"""NATS-based rollout function for miles <-> AREnABase integration.

Uses a global background worker (same pattern as fully_async_rollout.py)
that continuously publishes tasks to NATS and collects results. The main
``generate_rollout`` function just drains completed sample groups from
the output queue.

Sample conversion (``full_trajectory``): the entire multi-turn conversation
becomes one Sample. When the gym used the GenerateClient, real token-level
data (token_ids/loss_mask/log_probs) flows through directly; otherwise the
conversation is tokenised locally via ``MultiTurnLossMaskGenerator``. Under
``--arena-train-segments all`` a compacted episode (several self-contained
``steps`` with ``segment_end`` markers) becomes one Sample per segment; the
segments share one ``rollout_id`` and one reward (ADR-0011).

Wired in via ``--rollout-function-path miles_plugins.arena.nats_arena.nats_rollout.generate_rollout``.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from miles.utils.types import Sample

from miles_plugins.arena.nats_arena.message_format import (
    SALVAGEABLE_RESULT_STATUSES,
    extract_trajectories,
    parse_result,
    sample_to_task,
    serialize_task,
)
from miles_plugins.arena.nats_arena.mixture_controller import MixtureController
from miles_plugins.arena.nats_arena.stream_config import results_stream_config

# Lazy-imported in _maybe_build_autoscaler / _maybe_build_timing_tracker
# so that import errors in these optional modules don't prevent
# generate_rollout from being importable (which kills the whole training
# job with a confusing "no attribute 'generate_rollout'" error).
GymAutoscaler = None  # type: ignore[assignment]
RolloutTimingTracker = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = ["generate_rollout"]

# Agent-loop terminal states (AREnABase vulcan_agent AgentResult.stop_reason)
# that indicate a degenerate / incomplete trajectory which must NOT be trained
# on as a clean COMPLETED rollout. Distinct from the per-turn SGLang
# finish_reason ("stop"/"length"/"abort"): e.g. a trajectory whose context
# overflowed 131k and failed drop-oldest recovery terminates with
# ``context_error`` while its last turn's finish_reason is "stop". Flagged
# TRUNCATED -> remove_sample=True so their tokens are masked out of the loss
# (a final turn clipped by the per-turn cap can instead be partially salvaged
# via --arena-mask-clipped-final-turn; degenerate stops never are -- with ONE
# opt-in exception: ``timeout`` is the Harbor agent's per-task wall-clock
# deadline (AgentTimeoutError), whose recorded turns all ended cleanly and
# whose verifier still ran, so it can be kept via
# --arena-keep-timeout-trajectories).
# ``completed`` and ``max_iterations`` are intentionally NOT here (legitimate
# finishes).
_DEGENERATE_AGENT_STOP = frozenset({
    "context_error",
    "context_churn",
    "empty_response",
    "max_budget",
    "timeout",
    "no_choices",
})



# ---------------------------------------------------------------------------
# Global worker (persists across generate_rollout calls within the
# same RolloutManager Ray actor — same pattern as fully_async_rollout.py)
# ---------------------------------------------------------------------------
_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_source):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("Creating new global NATS rollout worker...")
            _global_worker = NATSRolloutWorker(args, data_source)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


# ---------------------------------------------------------------------------
# NATS connection helpers (used inside the worker's async event loop)
# ---------------------------------------------------------------------------

def _get_nats_config() -> dict[str, str]:
    return {
        "url": os.environ.get("NATS_URL", "nats://nats-svc:4222"),
        "tasks_stream": os.environ.get("NATS_TASKS_STREAM", "ARENA_TASKS"),
        "tasks_subject_prefix": os.environ.get("NATS_TASKS_SUBJECT_PREFIX", "arena.tasks"),
        "default_gym": os.environ.get("ARENA_DEFAULT_GYM", ""),
        "results_stream": os.environ.get("NATS_RESULTS_STREAM", "ARENA_RESULTS"),
        "results_subject": os.environ.get("NATS_RESULTS_SUBJECT", "arena.results"),
        "results_consumer": os.environ.get("NATS_RESULTS_CONSUMER", "slime-trainer"),
    }


_NATS_RECONNECT_ERRORS = (
    "consumer not found",
    "consumer deleted",
    "stream not found",
    "connection closed",
    "connection lost",
    "connection reset",
    "stale connection",
    "no responders",
    # JetStream returns 503 ServiceUnavailable when the durable consumer or
    # stream backing a pull-subscription has gone away (e.g. NATS pod was
    # rescheduled and lost in-memory state). Treat as reconnect-recoverable
    # so we re-run _ensure_nats and recreate the missing assets.
    "serviceunavailable",
    "service unavailable",
    "503",
    "jetstream not enabled",
    "jetstream not available",
    "leadership change",
    "no leader",
)

_ALLOWED_KEYS = {"role", "content", "tool_calls", "tool_call_id",
                 "name", "reasoning_content", "tools"}


def _is_nats_reconnect_error(exc: Exception) -> bool:
    # Class-based check first — survives changes in error message wording.
    try:
        from nats.errors import (
            ConnectionClosedError,
            NoRespondersError,
            StaleConnectionError,
            TimeoutError as NatsTimeoutError,
        )
        from nats.js.errors import (
            APIError,
            BadRequestError,
            ConsumerNotFoundError,
            NotFoundError,
            ServiceUnavailableError,
            StreamNotFoundError,
        )
    except ImportError:
        ConnectionClosedError = NoRespondersError = StaleConnectionError = ()  # type: ignore
        NatsTimeoutError = ()  # type: ignore
        ServiceUnavailableError = ConsumerNotFoundError = StreamNotFoundError = ()  # type: ignore
        NotFoundError = APIError = BadRequestError = ()  # type: ignore

    if isinstance(
        exc,
        (
            ServiceUnavailableError,
            ConsumerNotFoundError,
            StreamNotFoundError,
            NotFoundError,
            ConnectionClosedError,
            NoRespondersError,
            StaleConnectionError,
        ),
    ):
        return True

    # Fallback string-match for error wrappers that don't preserve class.
    err_str = str(exc).lower()
    return any(marker in err_str for marker in _NATS_RECONNECT_ERRORS)


async def _ensure_nats(cfg, existing_nc=None, *, purge_on_init=True, is_reconnect=False):
    """Connect to NATS and ensure streams/consumers exist.

    On initial connect (``is_reconnect=False``), purges any stale streams so
    a fresh training run starts from a clean slate.

    On reconnect (``is_reconnect=True``), preserves existing stream contents
    and only creates streams/consumers that have gone missing — so messages
    that the NATS server still has aren't thrown away.
    """
    import nats
    from nats.js.api import ConsumerConfig, RetentionPolicy, StreamConfig

    if existing_nc is not None and not existing_nc.is_closed:
        try:
            await existing_nc.close()
        except Exception:
            pass

    nc = await nats.connect(
        cfg["url"],
        max_reconnect_attempts=-1,
        reconnect_time_wait=2,
        ping_interval=20,
        max_outstanding_pings=5,
        connect_timeout=30,
        allow_reconnect=True,
    )
    js = nc.jetstream()

    # Tasks stream: wildcard subject to support per-gym routing.
    # arena.tasks.* matches arena.tasks.collinear_erp_hr, arena.tasks.skillsbench, etc.
    tasks_wildcard = cfg["tasks_subject_prefix"] + ".*"
    try:
        info = await js.stream_info(cfg["tasks_stream"])
        if not is_reconnect and purge_on_init:
            await js.purge_stream(cfg["tasks_stream"])
            logger.info("Purged stale tasks stream: %s", cfg["tasks_stream"])
        if tasks_wildcard not in (info.config.subjects or []):
            await js.update_stream(config=StreamConfig(
                name=cfg["tasks_stream"],
                subjects=[tasks_wildcard],
                retention=RetentionPolicy.WORK_QUEUE,
            ))
            logger.info("Updated tasks stream subjects to: %s", tasks_wildcard)
    except Exception:
        await js.add_stream(config=StreamConfig(
            name=cfg["tasks_stream"],
            subjects=[tasks_wildcard],
            retention=RetentionPolicy.WORK_QUEUE,
        ))
        logger.info(
            "%s tasks stream: %s",
            "Recreated (was missing after reconnect)" if is_reconnect else "Created",
            cfg["tasks_stream"],
        )

    # Results stream: single shared subject (all gyms publish here). Bounded
    # (max_age + max_bytes, discard=OLD) so a stalled trainer consumer never
    # fills the JetStream volume — the eviction that lost glm53 r6.
    results_cfg = results_stream_config(cfg["results_stream"], cfg["results_subject"])
    try:
        await js.stream_info(cfg["results_stream"])
        stream_exists = True
    except Exception:
        stream_exists = False
    if stream_exists:
        # Apply the current bounds to a pre-existing stream so a run started
        # before the bounds existed cannot fill the volume. update_stream only
        # mutates max_age/max_bytes/discard here; retention and subjects are
        # unchanged, so it never rejects.
        try:
            await js.update_stream(config=results_cfg)
        except Exception:
            logger.warning(
                "Could not apply results-stream bounds via update_stream; "
                "leaving the existing config in place.",
                exc_info=True,
            )
        if not is_reconnect and purge_on_init:
            await js.purge_stream(cfg["results_stream"])
            logger.info("Purged stale results stream: %s", cfg["results_stream"])
    else:
        await js.add_stream(config=results_cfg)
        logger.info(
            "%s results stream: %s",
            "Recreated (was missing after reconnect)" if is_reconnect else "Created",
            cfg["results_stream"],
        )

    try:
        await js.consumer_info(cfg["results_stream"], cfg["results_consumer"])
    except Exception:
        await js.add_consumer(
            stream=cfg["results_stream"],
            config=ConsumerConfig(
                durable_name=cfg["results_consumer"],
                ack_policy="explicit",
                max_deliver=3,
                ack_wait=3600,
            ),
        )
        logger.info(
            "%s results consumer: %s",
            "Recreated (was missing after reconnect)" if is_reconnect else "Created",
            cfg["results_consumer"],
        )

    psub = await js.pull_subscribe(
        subject=cfg["results_subject"],
        durable=cfg["results_consumer"],
    )
    logger.info(
        "NATS connected (%s): tasks=%s (stream=%s), results=%s",
        "reconnect" if is_reconnect else "initial",
        tasks_wildcard, cfg["tasks_stream"], cfg["results_subject"],
    )
    return nc, js, psub


# ---------------------------------------------------------------------------
# Sample conversion — full_trajectory mode
# ---------------------------------------------------------------------------

# ``index`` stride between the segments of one episode under
# ``--arena-train-segments all``: ``index = gid * n + e + k * STRIDE``. 2**40
# leaves 2**23 episode ids below the stride, so the packed int64 stays exact,
# and ``k == 0`` reproduces today's ``gid * n + e`` so a single-segment
# episode keeps its exact identity in both modes (ADR-0011).
_SEGMENT_INDEX_STRIDE = 1 << 40


def _train_segments_mode(args) -> str:
    """Return the ``--arena-train-segments`` value; ``final`` when the hook never ran."""
    return str(getattr(args, "arena_train_segments", None) or "final")


def _training_steps(steps: list[dict], args) -> list[dict]:
    """Select the trajectory steps that become training samples.

    ``final`` (default) trains ``steps[-1]`` only, exactly as before. ``all``
    trains one Sample per step, but ONLY when at least one step carries the
    explicit ``segment_end`` marker that the Harbor gym stamps on an archived
    compaction segment (AREnATasks ADR-0048). The Inspect ``sglang_perstep``
    provider ships cumulative per-turn steps with ``has_generate_tokens`` on
    every step and no marker, so it stays on the ``steps[-1]`` path in both
    modes (ADR-0011).
    """
    if _train_segments_mode(args) == "all" and any(st.get("segment_end") for st in steps):
        return [st for st in steps if st.get("has_generate_tokens")]
    return steps[-1:]


@dataclass(frozen=True)
class _EpisodeContext:
    """Trajectory-level inputs of the per-step Sample conversion.

    One instance per trajectory. Every segment of a multi-segment episode is
    converted against the same context, so reward, truncation and the salvage
    flags stay EPISODE properties (ADR-0011).
    """

    task_id: str
    gym_name: str
    reward: float
    agent_stop_reason: str
    agent_degenerate: bool
    timeout_salvageable: bool
    context_salvageable: bool
    any_step_truncated: bool
    max_ctx: int | None
    args: Any
    weight_versions: list[str]

    @property
    def status(self) -> Sample.Status:
        """Episode status before the per-step overflow and log-prob checks."""
        if self.any_step_truncated or self.agent_degenerate:
            return Sample.Status.TRUNCATED
        return Sample.Status.COMPLETED


def _finish_sample(
    s: Sample,
    ctx: _EpisodeContext,
    *,
    removal_reason: str | None,
    kept_timeout: bool,
    kept_context_error: bool,
) -> Sample:
    """Stamp the trajectory-level fields shared by the fast and slow paths."""
    # Per-turn SGLang weight versions for the off_policy_round metric: the
    # trajectory list on every segment, no per-segment slicing (ADR-0011).
    s.weight_versions = list(ctx.weight_versions)
    # segment / n_segments land on EVERY sample (ADR-0011). The default is a
    # one-sample episode, which is what the slow path emits; the fast path
    # overwrites both for a multi-segment episode.
    s.metadata = {
        "task_id": ctx.task_id,
        "mode": "full_trajectory",
        "gym_name": ctx.gym_name,
        "segment": 0,
        "n_segments": 1,
    }
    # Removal-reason bookkeeping for the per-rollout "Removal reasons"
    # summary (generate_rollout). agent_stop_reason is recorded for every
    # real sample; removal_reason only when remove_sample was set;
    # kept_timeout only for --arena-keep-timeout-trajectories salvages.
    if ctx.agent_stop_reason:
        s.metadata["agent_stop_reason"] = ctx.agent_stop_reason
    if removal_reason:
        s.metadata["removal_reason"] = removal_reason
    if kept_timeout:
        s.metadata["kept_timeout"] = True
    if kept_context_error:
        s.metadata["kept_context_error"] = True
    return s


def _step_to_sample(step: dict, ctx: _EpisodeContext, *, is_final: bool) -> Sample:
    """Convert one GenerateClient step (cumulative token arrays) into a Sample.

    ``is_final`` marks the segment that produced the episode end. Only that
    segment can carry a clipped final turn (an archived segment ends on a
    completed turn, AREnATasks ADR-0048), so ``--arena-mask-clipped-final-turn``
    zeroes a trailing 1-run there only; an earlier segment has nothing to mask
    and stays trainable. Hard context overflow and the rollout-logprob check
    are per step.
    """
    task_id = ctx.task_id
    args = ctx.args
    max_ctx = ctx.max_ctx
    any_step_truncated = ctx.any_step_truncated
    agent_degenerate = ctx.agent_degenerate
    agent_stop_reason = ctx.agent_stop_reason
    timeout_salvageable = ctx.timeout_salvageable
    stop_salvageable = ctx.timeout_salvageable or ctx.context_salvageable
    # True when the agent stop alone makes the sample untrainable.
    agent_blocks_training = agent_degenerate and not stop_salvageable
    status = ctx.status
    # Why the sample was dropped from the loss (None when trained on);
    # stamped into Sample.metadata for the per-rollout "Removal reasons"
    # summary log. Set alongside every ``remove_sample = True`` below.
    removal_reason = None
    kept_timeout = False
    kept_context_error = False
    s = Sample()
    token_ids = step["token_ids"]
    loss_mask = step["loss_mask"]
    log_probs = step.get("log_probs", [])
    # miles convention (unchanged from slime): response spans from the
    # FIRST loss-masked token to the end (interspersed tool/user 0s
    # stay in the mask), NOT the count of 1s. response_length,
    # loss_mask, and rollout_log_probs must all be aligned to this
    # same [first_1 : end] window so the CP slicing in the actor /
    # get_sum_of_sample_mean stays consistent.
    prompt_len = 0
    for m in loss_mask:
        if m == 0:
            prompt_len += 1
        else:
            break
    has_response = any(m == 1 for m in loss_mask)
    resp_loss_mask = loss_mask[prompt_len:] if has_response else []
    response_length = len(resp_loss_mask)
    resp_log_probs = log_probs[prompt_len:] if has_response and log_probs else []

    hard_overflow = False
    if max_ctx and len(token_ids) > max_ctx:
        hard_overflow = True
        orig_len = len(token_ids)
        token_ids = token_ids[:max_ctx]
        loss_mask = loss_mask[:max_ctx]
        if log_probs:
            log_probs = log_probs[:max_ctx]
        logger.warning(
            "Task %s: fast-path context overflow, hard-truncated %d -> %d tokens.",
            task_id, orig_len, len(token_ids),
        )
        prompt_len = 0
        for m in loss_mask:
            if m == 0:
                prompt_len += 1
            else:
                break
        has_response = any(m == 1 for m in loss_mask)
        resp_loss_mask = loss_mask[prompt_len:] if has_response else []
        response_length = len(resp_loss_mask)
        resp_log_probs = log_probs[prompt_len:] if has_response and log_probs else []
        status = Sample.Status.TRUNCATED

    s.tokens = token_ids
    s.loss_mask = resp_loss_mask
    s.response_length = response_length
    s.reward = ctx.reward

    # rollout_log_probs MUST be aligned 1:1 with the response window when
    # use_rollout_logprobs is on — the actor slices it with CP and
    # asserts len == response_length. A trajectory whose final
    # /generate was rejected (e.g. context-length overflow) comes back
    # with no token logprobs, so resp_log_probs is empty / mismatched.
    # Zero-fill to response_length so the CP slicer can't crash, and drop
    # the sample from training (remove_sample) since its logprobs are not
    # real. Padding keeps the group's n_samples count intact.
    bad_logprobs = response_length > 0 and len(resp_log_probs) != response_length
    if bad_logprobs:
        logger.warning(
            "Task %s: rollout_log_probs length %d != response_length %d "
            "(likely a rejected/overflow generate call); zero-filling and "
            "removing sample from training.",
            task_id, len(resp_log_probs), response_length,
        )
        resp_log_probs = [0.0] * response_length
        status = Sample.Status.ABORTED
    # Always emit a list (never None) so the actor's CP slicer, which
    # does len(rollout_log_probs), never sees None. Empty response →
    # empty list, which aligns with response_length==0.
    s.rollout_log_probs = resp_log_probs
    s.status = status
    # --arena-mask-clipped-final-turn: the gym ships ONE step whose
    # stop_reason is "length" exactly when the FINAL generate hit the
    # per-turn cap, and that final turn's output tokens are the
    # trailing run of 1s in the cumulative loss_mask (no prompt/tool
    # tokens follow the last generate). Zero only that run and keep
    # the earlier, cleanly-stopped turns trainable — instead of the
    # r5-lineage default of removing the whole sample. Degenerate
    # agent stops (except a "timeout" admitted by
    # --arena-keep-timeout-trajectories), hard context overflows, and
    # zero-filled logprobs are NOT salvageable and keep the old
    # removal behavior. Status stays TRUNCATED so truncated_ratio
    # remains an honest metric.
    final_clip_masked = False
    if (
        getattr(args, "arena_mask_clipped_final_turn", False)
        and status == Sample.Status.TRUNCATED
        and any_step_truncated
        and not agent_blocks_training
        and not hard_overflow
        and not bad_logprobs
    ):
        if not is_final:
            # An archived segment closed on a completed turn: the clipped
            # tokens live in the FINAL segment only, so there is nothing to
            # mask here and the segment stays trainable (ADR-0011).
            final_clip_masked = has_response
        else:
            tail = len(resp_loss_mask)
            while tail > 0 and resp_loss_mask[tail - 1] == 1:
                tail -= 1
            clipped_len = len(resp_loss_mask) - tail
            if clipped_len and any(m == 1 for m in resp_loss_mask[:tail]):
                s.loss_mask = resp_loss_mask[:tail] + [0] * clipped_len
                final_clip_masked = True
                logger.info(
                    "Task %s: masked clipped final turn (%d of %d response "
                    "tokens); earlier turns stay trainable.",
                    task_id, clipped_len, len(resp_loss_mask),
                )
    # --arena-keep-timeout-trajectories: keep the sample, loss_mask
    # untouched, when the ONLY thing flagging it TRUNCATED is the
    # agent-loop "timeout" stop -- no per-turn length clip, no hard
    # context overflow, no zero-filled logprobs, and at least one
    # trainable token. Precedence when a timeout trajectory ALSO has
    # a clipped final turn (any_step_truncated): the
    # --arena-mask-clipped-final-turn salvage above applies if that
    # flag is on (the timeout no longer blocks it); otherwise the
    # sample stays removed -- this branch never keeps a trajectory
    # with a clipped turn. Slow-path (messages-only) trajectories are
    # out of scope, exactly like the mask-clipped salvage.
    if (
        stop_salvageable
        and status == Sample.Status.TRUNCATED
        and not any_step_truncated
        and not hard_overflow
        and not bad_logprobs
        and has_response
    ):
        if timeout_salvageable:
            kept_timeout = True
        else:
            kept_context_error = True
        # #turns == number of 1-runs in the response loss_mask (each
        # assistant turn is one run, separated by tool/user 0s).
        n_turns = sum(
            1 for i, m in enumerate(resp_loss_mask)
            if m == 1 and (i == 0 or resp_loss_mask[i - 1] == 0)
        )
        logger.info(
            "Task %s: kept %s trajectory (%d turns, %d response "
            "tokens); agent %s but every turn ended cleanly.",
            task_id,
            "timeout" if kept_timeout else "context_error",
            n_turns, len(resp_loss_mask),
            "hit its wall-clock deadline" if kept_timeout
            else "ran out of context window",
        )
    if (
        status == Sample.Status.TRUNCATED
        and not final_clip_masked
        and not kept_timeout
        and not kept_context_error
    ) or bad_logprobs:
        s.remove_sample = True
        removal_reason = (
            "bad_logprobs" if bad_logprobs
            else "context_overflow" if hard_overflow
            else agent_stop_reason if agent_degenerate
            else "length" if any_step_truncated
            else "other"
        )
    return _finish_sample(
        s, ctx,
        removal_reason=removal_reason,
        kept_timeout=kept_timeout,
        kept_context_error=kept_context_error,
    )


def _messages_to_sample(messages: list[dict], ctx: _EpisodeContext, tokenizer) -> Sample:
    """Slow path: re-tokenise a messages-only trajectory into one Sample."""
    from miles.utils.mask_utils import MultiTurnLossMaskGenerator

    task_id = ctx.task_id
    args = ctx.args
    status = ctx.status
    removal_reason = None
    s = Sample()
    clean_messages = []
    for msg in messages:
        clean = {k: v for k, v in msg.items() if k in _ALLOWED_KEYS and v is not None}
        if clean.get("content") is None:
            clean["content"] = ""
        if (
            clean.get("role") == "assistant"
            and isinstance(clean.get("tool_calls"), list)
        ):
            for tc in clean["tool_calls"]:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        fn["arguments"] = {}
        clean_messages.append(clean)

    mask_generator = MultiTurnLossMaskGenerator(
        tokenizer, tokenizer_type=getattr(args, "loss_mask_type", None)
    )

    try:
        token_ids, loss_mask = mask_generator.get_loss_mask(clean_messages)
    except Exception as exc:
        logger.error(
            "Task %s: get_loss_mask failed: %s. Last msg: %s",
            task_id, exc, json.dumps(clean_messages[-1], default=str)[:2000],
        )
        raise

    slow_overflow = False
    if ctx.max_ctx and len(token_ids) > ctx.max_ctx:
        slow_overflow = True
        # Context overflow: hard-truncate token_ids/loss_mask (and any
        # parallel per-token field) to max_ctx. We don't bother dropping
        # the final assistant turn to keep the conversation well-formed —
        # an over-context sample is marked TRUNCATED below, which sets
        # remove_sample=True, so it's excluded from training anyway. A
        # simple clip keeps the arrays length-consistent for the actor's
        # CP slicer without the extra re-tokenize round-trip.
        orig_len = len(token_ids)
        token_ids = token_ids[:ctx.max_ctx]
        loss_mask = loss_mask[:ctx.max_ctx]
        logger.warning(
            "Task %s: context overflow, hard-truncated %d -> %d tokens "
            "(remove_sample via TRUNCATED status).",
            task_id, orig_len, len(token_ids),
        )
        status = Sample.Status.TRUNCATED

    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    resp_loss_mask = loss_mask[-response_length:] if response_length > 0 else []

    s.tokens = token_ids
    s.loss_mask = resp_loss_mask
    s.response_length = response_length
    s.reward = ctx.reward

    # The slow path re-tokenizes messages, so there are no token-level
    # logprobs. Under use_rollout_logprobs, convert_samples_to_train_data
    # gates the whole batch on samples[0].rollout_log_probs
    # (train_data_conversion.py:114-115): a group mixing fast- and
    # slow-path trajectories would carry a None row that crashes
    # tensorization (or, slow-first, silently drops the field for the
    # batch). Mirror the fast path's bad-logprobs handling: zero-fill to
    # response_length so the actor's CP slicer stays aligned, and drop
    # the sample from training since its logprobs are not real.
    if getattr(args, "use_rollout_logprobs", False):
        logger.warning(
            "Task %s: slow-path (messages-only) trajectory has no "
            "rollout_log_probs under use_rollout_logprobs; zero-filling "
            "and removing sample from training.",
            task_id,
        )
        s.rollout_log_probs = [0.0] * response_length
        status = Sample.Status.ABORTED
        s.remove_sample = True
        removal_reason = "no_logprobs"
    s.status = status
    if status == Sample.Status.TRUNCATED:
        s.remove_sample = True
        removal_reason = (
            "context_overflow" if slow_overflow
            else ctx.agent_stop_reason if ctx.agent_degenerate
            else "length" if ctx.any_step_truncated
            else "other"
        )
    return _finish_sample(
        s, ctx, removal_reason=removal_reason, kept_timeout=False, kept_context_error=False,
    )


def _result_to_episodes_full_trajectory(
    result: dict[str, Any],
    tokenizer,
    args,
) -> list[list[Sample]]:
    """Convert one result envelope into episodes: one ``list[Sample]`` per trajectory.

    A single-step trajectory, or any trajectory under ``--arena-train-segments
    final``, yields a one-sample episode (today's shape). Under ``all`` a
    trajectory whose ``steps`` carry ``segment_end`` markers yields one Sample
    per segment; the segments share the episode reward and, once
    ``_process_group`` stamps them, one ``rollout_id`` (ADR-0011).
    ``group_metrics`` lands on the first sample of the first episode only.
    """
    episodes: list[list[Sample]] = []
    task_id = result.get("task_id", "unknown")
    gym_name = result.get("gym_name", "unknown")
    # Per-task group_metrics dict computed gym-side. Stored on the FIRST
    # sample of the group so the trainer's log_rollout_data can pick it up
    # without double-counting. None for older gym builds that don't emit it.
    group_metrics = result.get("group_metrics")
    trajectories = extract_trajectories(result)
    max_ctx = getattr(args, "rollout_max_context_len", None) or getattr(args, "sglang_context_length", None) or getattr(args, "max_tokens_per_gpu", None)

    for traj in trajectories:
        # --- Synthetic-failure path (client-agnostic) ---
        # The gym pads the trajectory list with synthetic 0-reward entries
        # when a sample never produced a real rollout: a fork-pool worker
        # that raised, or one that the dispatcher force-killed at its
        # per-task deadline (the straggler-bound fix). These carry an
        # explicit ``synthetic`` flag (newer gym builds) and/or
        # ``stop_reason == "errored"`` with empty messages/steps.
        #
        # We must NOT build a Sample for these here. A synthetic entry has no
        # tokens, so the Sample would carry tokens=[] / response_length=0.
        # ``remove_sample`` only zeros the loss_mask
        # (train_data_conversion.py:98-99) and ``Status.FAILED`` is not
        # filtered anywhere — neither drops the row — so a zero-length sample
        # still gets packed into train_data, where prompt_length =
        # total_length - response_length = 0 makes the Megatron loss / CP
        # slicing degenerate. Instead we SKIP the trajectory and let
        # ``_process_group``'s existing pad loop fill the slot by copying a
        # real sibling's tokens with remove_sample=True (the proven path that
        # already handles rollout_log_probs alignment). The resulting shortfall
        # is what the failed-fraction metric counts. Works identically for
        # generate and openai clients since it keys on the flag, not tokens.
        steps = traj.get("steps") or []
        messages = traj.get("messages")
        reward = traj.get("reward") or 0.0
        is_synthetic = bool(traj.get("synthetic")) or (
            str(traj.get("stop_reason") or "").lower() == "errored"
            and not messages and not steps
        )
        if is_synthetic:
            logger.debug(
                "Task %s: skipping synthetic/failed trajectory; "
                "_process_group will pad the slot with a sibling copy.",
                task_id,
            )
            continue

        # Mark TRUNCATED when EITHER the last turn or any intermediate turn
        # hit ``finish_reason="length"`` — same logic as the slow path. A
        # multi-turn agent often ends its terminal turn cleanly while an
        # earlier turn truncated mid-tool-call; flagging only the last step
        # would under-count truncated_ratio and silently reward cut-off
        # actions. An archived compaction segment carries no stop_reason
        # (AREnATasks ADR-0048), so only the final segment can set this and
        # truncation stays an episode property (ADR-0011).
        any_step_truncated = any(
            "length" in str(st.get("stop_reason") or "").lower() for st in steps
        )
        # Agent-loop degenerate terminal states (context_error, context_churn,
        # empty_response, max_budget, timeout, no_choices) are NOT length-
        # truncations — the per-turn stop_reason is "stop" — so the check above
        # misses them. The trajectory is a mangled/incomplete conversation (e.g.
        # context-overflow recovery failed) that must not be trained on as
        # COMPLETED. Flag TRUNCATED so remove_sample fires below.
        agent_stop_reason = str(traj.get("agent_stop_reason") or "").lower()
        agent_degenerate = agent_stop_reason in _DEGENERATE_AGENT_STOP
        # --arena-keep-timeout-trajectories: the Harbor gym worker maps its
        # per-task wall-clock AgentTimeoutError to agent_stop_reason
        # "timeout" even though every recorded turn ended with finish_reason
        # "stop", the token stream ends at the model's turn-close token (no
        # partial turn) and the verifier still ran, so the reward is real.
        # With the flag on, a "timeout" stop no longer disqualifies the
        # sample on its own (nor blocks --arena-mask-clipped-final-turn
        # salvage); every other degenerate stop is unaffected. Status stays
        # TRUNCATED either way so truncated_ratio remains an honest metric.
        timeout_salvageable = (
            bool(getattr(args, "arena_keep_timeout_trajectories", False))
            and agent_stop_reason == "timeout"
        )
        # --arena-keep-context-error-trajectories: same shape for
        # "context_error". The arena Terminus-2 agent ends the episode when
        # the engine rejects the NEXT turn's prompt as too long; the recorded
        # turns all ended cleanly, the token stream is the complete episode
        # up to that point and the verifier still ran. Status stays TRUNCATED.
        context_salvageable = (
            bool(getattr(args, "arena_keep_context_error_trajectories", False))
            and agent_stop_reason == "context_error"
        )
        # Per-turn SGLang weight versions for the off_policy_round metric.
        # Prefer the explicit parallel list emitted by the gym; fall back to
        # the per-step ``weight_version`` keys for older gym builds. Empty
        # list (no version reported) → metric treats the sample as untagged.
        wvs = traj.get("weight_versions")
        if not wvs:
            wvs = [st.get("weight_version") for st in steps]
        ctx = _EpisodeContext(
            task_id=task_id,
            gym_name=gym_name,
            reward=reward,
            agent_stop_reason=agent_stop_reason,
            agent_degenerate=agent_degenerate,
            timeout_salvageable=timeout_salvageable,
            context_salvageable=context_salvageable,
            any_step_truncated=any_step_truncated,
            max_ctx=max_ctx,
            args=args,
            weight_versions=[str(v) for v in wvs if v is not None],
        )

        training = _training_steps(steps, args)
        # --- Fast path: GenerateClient produced real token-level data ---
        # When the gym used ARENA_CLIENT_TYPE=generate, each training step has
        # cumulative token_ids/loss_mask/log_probs built from /generate
        # responses. Skip re-tokenization entirely.
        if training and training[-1].get("has_generate_tokens"):
            episode = [
                _step_to_sample(st, ctx, is_final=(k == len(training) - 1))
                for k, st in enumerate(training)
            ]
            for k, s in enumerate(episode):
                s.metadata["segment"] = k
                s.metadata["n_segments"] = len(training)
        else:
            if not messages:
                logger.warning("Trajectory for %s has no messages, skipping", task_id)
                continue
            episode = [_messages_to_sample(messages, ctx, tokenizer)]
        # Attach group_metrics to ONLY the first sample of each group —
        # all samples in the group share the same metrics, and stamping
        # them on every sample would inflate the per-batch aggregate.
        if group_metrics is not None and not episodes:
            episode[0].metadata["group_metrics"] = group_metrics
        episodes.append(episode)
    return episodes


def _result_to_samples_full_trajectory(
    result: dict[str, Any],
    tokenizer,
    args,
) -> list[Sample]:
    """Flat view of ``_result_to_episodes_full_trajectory`` (signature unchanged)."""
    return [
        s
        for episode in _result_to_episodes_full_trajectory(result, tokenizer, args)
        for s in episode
    ]


def _publisher_max_in_flight(args) -> int:
    """In-flight publisher cap: ``arena_inflight_multiplier`` x ``rollout_batch_size``.

    Pure function of ``args`` so the cap is unit-testable without NATS. The
    getattr default (2) matches the argparse default and keeps every existing
    run byte-identical.
    """
    return int(getattr(args, "arena_inflight_multiplier", 2)) * int(args.rollout_batch_size)


def _episode_key(s: Sample) -> tuple[int | None, int | None]:
    """Episode identity: ``rollout_id`` when stamped (``all`` mode), else ``index``."""
    return (s.group_index, s.rollout_id if s.rollout_id is not None else s.index)


def _episodes(groups: list[list[Sample]]) -> list[list[Sample]]:
    """Group the samples of ``groups`` into episodes, in batch order (ADR-0011).

    Under ``--arena-train-segments final`` every sample is its own episode, so
    every number derived from this is unchanged. Under ``all`` the segments of
    one episode share a ``rollout_id`` and land in one list, first segment first.
    """
    by_key: dict[tuple[int | None, int | None], list[Sample]] = {}
    for g in groups:
        for s in g:
            by_key.setdefault(_episode_key(s), []).append(s)
    return list(by_key.values())


def _episode_representatives(groups: list[list[Sample]]) -> list[Sample]:
    """Return one Sample per episode: its first segment.

    ``reward`` and ``weight_versions`` are identical on every segment, and the
    first segment of a group's first episode carries ``group_metrics``, so
    episode-level reductions of those read this sample. ``remove_sample``,
    ``removal_reason``, ``status`` and the ``kept_*`` flags are per segment
    (``hard_overflow``, ``bad_logprobs``, a final segment that is one clipped
    turn): reduce them over ``_episodes`` or read ``_removal_representative``.
    """
    return [e[0] for e in _episodes(groups)]


def _removal_representative(episode: list[Sample]) -> Sample:
    """The segment that speaks for the episode in the "Removal reasons" counts.

    A later segment can be dropped while the first one trains, so return the
    first removed segment, else the first segment.
    """
    return next((s for s in episode if getattr(s, "remove_sample", False)), episode[0])


def _episode_response_lengths(groups: list[list[Sample]]) -> list[int]:
    """Response tokens per episode: the sum over its segments."""
    return [
        sum(s.response_length for s in e if isinstance(getattr(s, "response_length", None), int))
        for e in _episodes(groups)
    ]


def _batch_telemetry(data: list[list[Sample]], all_data: list[list[Sample]]) -> dict[str, float]:
    """Per-rollout counts and reward-collapse scalars, counted per EPISODE.

    ``data`` is the post-filter batch that trains; ``all_data`` the pre-filter
    population (see ``generate_rollout``). A multi-segment episode (ADR-0011)
    is one unit everywhere: its reward is read once, it counts as removed or
    truncated when ANY segment is, and its response length is the sum over its
    segments. Sample-level counts (``rollout/num_training_samples``) come from
    miles ``log_rollout_data``, not from here.
    """
    data_episodes = _episodes(data)
    episodes = [e[0] for e in data_episodes]
    total_episodes = len(episodes)
    rewards = [s.reward for s in episodes if isinstance(s.reward, (int, float))]
    nonzero_groups = sum(1 for g in data if any(isinstance(s.reward, (int, float)) and s.reward > 0 for s in g))

    # ---- failed/dropped-sample telemetry ----------------------------------
    # What fraction of the GBS is excluded from training. Two distinct
    # buckets (kept separate so we can tell "gym couldn't produce a rollout"
    # apart from "rollout produced but unusable for training"):
    #   failed_frac   — episodes the gym padded as failures (fork-pool worker
    #       raised, or the dispatcher force-killed a straggler at its
    #       per-task deadline). mode == "failed".
    #   removed_frac  — ALL episodes with remove_sample=True, which also
    #       includes truncation / bad-logprob / context-overflow drops on
    #       top of the failed ones. This is the true "lost training signal"
    #       fraction; removed_frac >= failed_frac by construction.
    failed_count = sum(1 for s in episodes if (s.metadata or {}).get("mode") == "failed")
    # remove_sample and status are per segment: an episode is removed when ANY
    # segment is dropped (a final segment that is one clipped turn under
    # --arena-mask-clipped-final-turn, a per-segment bad_logprobs or
    # hard_overflow) and truncated when ANY segment is TRUNCATED. Under
    # ``final`` the episode is its single sample, so the numbers match.
    removed_count = sum(1 for e in data_episodes if any(getattr(s, "remove_sample", False) for s in e))
    all_data_episodes = _episodes(all_data)
    all_episodes = [e[0] for e in all_data_episodes]
    truncated_count = sum(
        1 for e in all_data_episodes if any(getattr(s, "status", None) == Sample.Status.TRUNCATED for s in e)
    )
    all_data_total = len(all_episodes)

    # ---- reward-collapse telemetry ----------------------------------------
    # We're trying to detect when GRPO advantages flatline (all-zero groups →
    # group_std=0 → advantage=0 → grad_norm=0). Five scalars on every rollout:
    #   - reward_nonzero_frac        : per-episode fraction with reward > 0
    #   - zero_reward_groups_frac    : fraction of groups where ALL episodes are 0
    #                                   (this is the kill metric: when it -> 1.0,
    #                                    advantage signal is gone)
    #   - group_std                  : mean of per-group reward std-dev
    #                                   (drops to 0 when groups become uniform)
    #   - reward_p25/p50/p75/p90     : reward distribution percentiles
    #   - avg_response_length        : mean response token count (catches model
    #                                   collapsing to trivial / one-tool outputs)
    #
    # These are computed over ``all_data`` (the pre-filter population), NOT
    # ``data``. The dynamic-sampling filter drops exactly the zero-variance /
    # all-zero groups these metrics are meant to count, so measuring them on the
    # post-filter batch would mask a genuine collapse (zero_reward_groups_frac
    # could never reach 1.0, group_std would be structurally inflated). When the
    # filter is off, ``all_data == data`` and behaviour is unchanged. This
    # mirrors sglang_rollout, which keeps ``all_data`` separate from ``data``.
    all_data_rewards = [s.reward for s in all_episodes if isinstance(s.reward, (int, float))]
    all_data_nonzero_count = sum(1 for r in all_data_rewards if r > 0)
    zero_reward_groups = sum(
        1 for g in all_data
        if all(
            (isinstance(s.reward, (int, float)) and s.reward == 0)
            or s.reward is None
            for s in g
        )
    )

    # Per-group std (group = one prompt, n_samples_per_prompt episodes).
    def _group_std(g: list[Sample]) -> float:
        rs = [s.reward for s in _episode_representatives([g]) if isinstance(s.reward, (int, float))]
        if len(rs) < 2:
            return 0.0
        m = sum(rs) / len(rs)
        return (sum((r - m) ** 2 for r in rs) / len(rs)) ** 0.5
    group_stds = [_group_std(g) for g in all_data]

    # Reward distribution percentiles.
    def _pct(xs: list[float], p: int) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        idx = min(int(len(ys) * p / 100), len(ys) - 1)
        return ys[idx]

    # Per-episode response length (loss_mask window), summed over segments.
    response_lens = _episode_response_lengths(all_data)

    return {
        "total_episodes": total_episodes,
        "avg_reward": sum(rewards) / max(total_episodes, 1),
        "nonzero_count": sum(1 for r in rewards if r > 0),
        "max_reward": max(rewards) if rewards else 0.0,
        "nonzero_groups": nonzero_groups,
        "failed_count": failed_count,
        "removed_count": removed_count,
        "truncated_count": truncated_count,
        "all_data_total": all_data_total,
        "failed_frac": failed_count / max(total_episodes, 1),
        "removed_frac": removed_count / max(total_episodes, 1),
        "truncated_ratio": truncated_count / max(all_data_total, 1),
        "reward_nonzero_frac": all_data_nonzero_count / max(all_data_total, 1),
        "zero_reward_groups_frac": zero_reward_groups / max(len(all_data), 1),
        "mean_group_std": sum(group_stds) / max(len(group_stds), 1),
        "reward_p25": _pct(all_data_rewards, 25),
        "reward_p50": _pct(all_data_rewards, 50),
        "reward_p75": _pct(all_data_rewards, 75),
        "reward_p90": _pct(all_data_rewards, 90),
        "avg_response_length": sum(response_lens) / max(len(response_lens), 1),
    }


def _removal_reason_counts(groups) -> tuple[dict[str, int], int, int]:
    """Aggregate why samples were removed from the loss, for the per-rollout
    "Removal reasons" summary log.

    Returns ``(reason -> count over samples with remove_sample=True,
    number of samples kept by --arena-keep-timeout-trajectories,
    number kept by --arena-keep-context-error-trajectories)``. Reasons
    come from ``Sample.metadata["removal_reason"]`` as stamped by
    ``_result_to_samples_full_trajectory``; gym-side pads (``mode="failed"``)
    count as ``failed``; anything else unattributed is ``other``.
    """
    removal_reasons: dict[str, int] = {}
    kept_timeout_count = 0
    kept_context_error_count = 0
    for g in groups:
        for s in g:
            meta = s.metadata or {}
            if meta.get("kept_timeout"):
                kept_timeout_count += 1
            if meta.get("kept_context_error"):
                kept_context_error_count += 1
            if not getattr(s, "remove_sample", False):
                continue
            reason = meta.get("removal_reason") or (
                "failed" if meta.get("mode") == "failed" else "other"
            )
            removal_reasons[reason] = removal_reasons.get(reason, 0) + 1
    return removal_reasons, kept_timeout_count, kept_context_error_count


# ---------------------------------------------------------------------------
# NATSRolloutWorker — background thread, same pattern as fully_async_rollout.py
# ---------------------------------------------------------------------------

class NATSRolloutWorker:
    """Background worker that continuously publishes tasks to NATS and
    collects results into an output queue.

    The main generate_rollout function just drains completed groups
    from ``output_queue``.
    """

    def __init__(self, args, data_source):
        import uuid

        self.args = args
        self.data_source = data_source
        self.data_source_lock = threading.Lock()
        self.running = True
        # Queue capacity expressed in groups, sized to hold ~10 GBS-worth
        # of samples. Each group is ``n_samples_per_prompt`` samples, so
        # total sample capacity = (10 * GBS / n_per_prompt) groups *
        # n_per_prompt samples = 10 * GBS samples. With GBS=256 and
        # n=16 that's 160 groups (= 10 training steps' buffer). The .put
        # call is blocking — when full, the publisher waits for the
        # trainer to drain rather than dropping data.
        self.output_queue: queue.Queue[list[Sample]] = queue.Queue(
            maxsize=max(1, (10 * args.global_batch_size) // max(1, args.n_samples_per_prompt))
        )
        self.worker_thread = None

        # Session token — uniquely identifies *this* trainer process's NATS
        # publish-side identity. Stamped onto every published task; gyms
        # echo it back in the result. The fetch loop drops any result
        # whose session != self.session.
        #
        # Why a session token (and not just task_id):
        #   task_id is the lakeFS instance_id, which is stable across
        #   restarts (same prompt → same id). After a checkpoint rollback
        #   (trainer crashed at step 15, resumed from step 10), the same
        #   prompts WILL get re-published with the same task_ids when the
        #   trainer cycles back through rollouts 11-14. Without a session
        #   token, an in-flight result from the *pre-crash* publish of
        #   that prompt could match `pending_expected[task_id]` on the
        #   *new* trainer and be silently accepted as the new attempt's
        #   result — training on a trajectory generated against weights
        #   that no longer exist.
        #
        # The session token is generated fresh on every NATSRolloutWorker
        # __init__ (so it changes on trainer process restart) but is held
        # constant across NATS reconnects within the same process (so we
        # don't drop our own legitimate in-flight results when the NATS
        # server flaps).
        self.session = uuid.uuid4().hex
        # Monotonic per-worker counter used to stamp per-group Sample
        # identities (shared ``group_index``, unique ``index``) on each
        # emitted prompt-group — see ``_process_group`` for the full
        # rationale. A single global worker (_global_worker) produces every
        # group, so this is process-unique.
        self._output_group_counter = 0
        logger.info("NATSRolloutWorker session token: %s", self.session)

        # Only ``full_trajectory`` is supported. The per_step path was removed
        # (it was broken/non-functional); reject any other value loudly so a
        # stale config can't silently fall through to unsupported behavior.
        self.sample_mode = getattr(args, "arena_sample_mode", None) or os.environ.get(
            "ARENA_SAMPLE_MODE", "full_trajectory"
        )
        if self.sample_mode != "full_trajectory":
            raise ValueError(
                f"Unsupported arena_sample_mode: {self.sample_mode!r}. "
                "Only 'full_trajectory' is supported (per_step was removed)."
            )

        self.n_per_prompt = args.n_samples_per_prompt
        self.concurrency = args.rollout_batch_size
        self.inflight_multiplier = int(getattr(args, "arena_inflight_multiplier", 2))
        self.max_in_flight = _publisher_max_in_flight(args)
        # ``--arena-train-segments``: ``final`` (today) or ``all`` (one Sample
        # per compaction segment, shared rollout_id; ADR-0011). Read by
        # generate_rollout's partial-batch guard.
        self._train_segments = _train_segments_mode(args)

        # Tokenizer is used by _result_to_samples_full_trajectory to tokenise
        # conversations locally when the gym did NOT use the GenerateClient
        # (i.e. no real token-level data attached to the trajectory).
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            args.hf_checkpoint, trust_remote_code=True,
        )

        self.mixture_controller: MixtureController | None = None
        mixture_targets = getattr(args, "gym_mixture_targets", None)
        if mixture_targets:
            if isinstance(mixture_targets, str):
                import json as _json
                mixture_targets = _json.loads(mixture_targets)
            self.mixture_controller = MixtureController(
                target_ratios=mixture_targets,
                adjustment_interval=getattr(args, "mixture_adjustment_interval", 60.0),
                smoothing=getattr(args, "mixture_smoothing", 0.3),
            )
            if hasattr(data_source, "get_restored_weights"):
                restored = data_source.get_restored_weights()
                if restored:
                    self.mixture_controller.load_state_dict(
                        {"current_weights": restored}
                    )
            logger.info(
                "MixtureController enabled: targets=%s, interval=%.0fs",
                mixture_targets,
                getattr(args, "mixture_adjustment_interval", 60.0),
            )

        self.timing_tracker: RolloutTimingTracker | None = self._maybe_build_timing_tracker(args)
        self.gym_autoscaler: GymAutoscaler | None = self._maybe_build_autoscaler(args)

    def _maybe_build_timing_tracker(self, args):
        # The tracker is only useful when autoscaling is enabled AND auto-tune
        # is on (or profile mode), since nothing else consumes its stats.
        auto_tune = getattr(args, "gym_autoscale_auto_tune", True)
        profile = getattr(args, "gym_autoscale_profile", False)
        if not getattr(args, "gym_autoscale", False):
            return None
        if not auto_tune and not profile:
            return None

        try:
            from miles_plugins.arena.nats_arena.rollout_timing_tracker import (
                RolloutTimingTracker as _RolloutTimingTracker,
            )
        except ImportError as exc:
            logger.warning(
                "rollout_timing_tracker module failed to import (%s). "
                "Tracker disabled; autoscaler will use static config values.",
                exc,
            )
            return None

        tracker = _RolloutTimingTracker(
            short_window_size=int(getattr(args, "gym_autoscale_short_window", 100)),
            long_window_size=int(getattr(args, "gym_autoscale_long_window", 1000)),
        )
        if hasattr(self.data_source, "get_restored_timing_tracker_state"):
            try:
                restored = self.data_source.get_restored_timing_tracker_state()
                if restored:
                    tracker.load_state_dict(restored)
            except Exception:
                logger.exception("Failed to restore RolloutTimingTracker state")
        logger.info(
            "RolloutTimingTracker enabled: short_window=%d, long_window=%d",
            tracker.short_size, tracker.long_size,
        )
        return tracker

    def _maybe_build_autoscaler(self, args):
        if not getattr(args, "gym_autoscale", False):
            return None

        try:
            from miles_plugins.arena.nats_arena.gym_autoscaler import (
                GymAutoscaler as _GymAutoscaler,
            )
        except ImportError as exc:
            logger.warning(
                "gym_autoscale is enabled but gym_autoscaler module failed to "
                "import (%s). Autoscaler disabled. Ensure gym_autoscaler.py "
                "and rollout_timing_tracker.py are synced to the code mount.",
                exc,
            )
            return None

        raw_config = getattr(args, "gym_autoscale_config", None)
        if raw_config is None:
            raw_config = os.environ.get("GYM_AUTOSCALE_CONFIG")
        if isinstance(raw_config, str):
            try:
                raw_config = json.loads(raw_config)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "gym_autoscale_config is not valid JSON; disabling autoscaler.",
                )
                return None
        if not raw_config:
            logger.warning(
                "--gym-autoscale is set but no gym_autoscale_config / "
                "GYM_AUTOSCALE_CONFIG provided; disabling autoscaler.",
            )
            return None

        gym_specs: dict[str, dict[str, Any]] = {}
        for gym_name, spec in raw_config.items():
            if "deploy_name" not in spec:
                logger.warning(
                    "gym_autoscale_config[%s] missing 'deploy_name'; skipping.",
                    gym_name,
                )
                continue
            initial = int(spec.get("initial_replicas", spec.get("replicas", 1)))
            gym_specs[gym_name] = {
                "deploy_name": spec["deploy_name"],
                "concurrency_per_pod": int(spec.get("concurrency_per_pod", 1)),
                "min_replicas": int(spec.get("min_replicas", max(1, initial))),
                "max_replicas": int(spec.get("max_replicas", max(initial * 4, initial + 1))),
                "initial_replicas": initial,
            }
        if not gym_specs:
            logger.warning("gym_autoscale_config produced no valid specs; disabling.")
            return None

        namespace = (
            getattr(args, "k8s_namespace", None)
            or os.environ.get("NAMESPACE")
            or "default"
        )
        tasks_stream = os.environ.get("NATS_TASKS_STREAM", "ARENA_TASKS")

        return _GymAutoscaler(
            gym_specs=gym_specs,
            namespace=namespace,
            tasks_stream=tasks_stream,
            consumer_prefix=os.environ.get("NATS_GYM_CONSUMER_PREFIX", "gym-worker"),
            mixture_controller=self.mixture_controller,
            timing_tracker=self.timing_tracker,
            interval_secs=float(getattr(args, "gym_autoscale_interval", 30.0)),
            cooldown_secs=float(getattr(args, "gym_autoscale_cooldown", 180.0)),
            warmup_secs=float(getattr(args, "gym_autoscale_warmup", 1800.0)),
            headroom=float(getattr(args, "gym_autoscale_headroom", 2.0)),
            deficit_alpha=float(getattr(args, "gym_autoscale_deficit_alpha", 0.5)),
            dry_run=bool(getattr(args, "gym_autoscale_dry_run", False)),
            auto_tune=bool(getattr(args, "gym_autoscale_auto_tune", True)),
            growth_threshold=float(getattr(args, "gym_autoscale_growth_threshold", 1.3)),
            growth_preemption=float(getattr(args, "gym_autoscale_growth_preemption", 1.25)),
            profile_mode=bool(getattr(args, "gym_autoscale_profile", False)),
            profile_duration_secs=float(getattr(args, "gym_autoscale_profile_duration", 1800.0)),
            profile_only=bool(getattr(args, "gym_autoscale_profile_only", False)),
            min_tracker_samples=int(getattr(args, "gym_autoscale_min_samples", 50)),
        )

    async def _worker_loop(self):
        cfg = _get_nats_config()
        _nc, js, psub = await _ensure_nats(cfg)

        # Load consumed instance_ids from checkpoint for resume skip logic.
        consumed_ids: set[str] = set()
        if hasattr(self.data_source, "get_consumed_instance_ids"):
            with self.data_source_lock:
                consumed_ids = self.data_source.get_consumed_instance_ids()
            if consumed_ids:
                logger.info("Resume: loaded %d consumed instance_ids to skip", len(consumed_ids))

        # Track how many tasks are in-flight (published but not yet collected).
        # With batch mode (1 msg per prompt), each in-flight task represents
        # one prompt that will return n_samples trajectories in a single result.
        in_flight = 0
        # Oversubscribe the in-flight pool to 2x rollout_batch_size so fast
        # trajectories keep flowing while long-tail multi-turn trajectories hold
        # their slots. With max_in_flight == gym capacity, a pod stalled between
        # turns (tool exec / grading) leaves SGLang KV idle (~50% observed);
        # 2x headroom lets a second trajectory keep the engine fed. The factor is
        # --arena-inflight-multiplier (default 2): on 32 engines 2x left KV at
        # ~0.83 with an empty queue (r10, 2026-09-07); raise it when engines
        # are under-fed.
        max_in_flight = self.max_in_flight
        # Map task_id -> list of results (1 result per task in batch mode)
        pending_results: dict[str, list[dict]] = {}
        # Map task_id -> expected count (always 1 in batch mode)
        pending_expected: dict[str, int] = {}
        # Map task_id -> wall-clock publish time (seconds since epoch).
        # Used by the DLQ sweep to time out tasks that NATS has finished
        # redelivering (max_deliver=3 attempts, ack_wait=3600s each) without
        # ever returning a result — i.e., a poison prompt that crashes every
        # gym worker that touches it. Without this sweep, those tasks live
        # in `pending_expected` forever and the rollout never completes.
        pending_publish_time: dict[str, float] = {}
        # Map unique tid -> (subject, serialized task bytes) for every in-flight
        # task. On a NATS reconnect that lost JetStream state (stream and
        # consumer recreated from scratch, e.g. after a volume-full eviction)
        # the trainer re-publishes these to the recreated tasks stream so the
        # gym workers re-run them under the same tid. Without this the tasks are
        # silently lost and the rollout hangs a few groups short (glm53 r6 lost
        # 2 of 64). Popped on completion and on DLQ expiry, alongside
        # pending_expected.
        pending_task_payload: dict[str, tuple[str, bytes]] = {}
        # Map unique tid -> raw instance_id (for resume consumed tracking).
        tid_to_instance_id: dict[str, str] = {}
        # Map unique tid -> dedup_key (instance_id+epoch) for resume guard.
        tid_to_dedup_key: dict[str, str] = {}
        group_counter = 0

        # DLQ deadline. NATS will retry up to `max_deliver=3` times, each
        # delivery has up to `ack_wait=3600s` to be acked. Worst-case
        # legitimate completion time is therefore 3*3600 = 10800s. We add a
        # generous safety margin (default 1800s) on top to allow for clock
        # skew, gym-side queueing, and slow rollouts that ran right up to
        # the ack_wait boundary. Tunable via NATS_TASK_DEADLINE_SECS env var
        # for gyms whose rollouts are reliably faster than 1h.
        task_deadline_secs = float(
            os.environ.get("NATS_TASK_DEADLINE_SECS", 3600 * 3 + 1800)
        )
        # How often to sweep for expired tasks. Cheap O(n) over
        # `pending_publish_time` entries; the loop tick is already ~1s, so
        # 30s is plenty.
        dlq_sweep_interval_secs = 30.0
        last_dlq_sweep = time.time()

        async def _reconnect_nats():
            """Reconnect to NATS after connection loss with exponential backoff.

            Preserves the in-flight book-keeping by default — the NATS server
            usually retains the work-queue messages and the durable consumer
            state. We only reset the in-flight counters when the durable
            consumer / results stream had to be recreated (i.e. JetStream
            actually lost state), to avoid waiting forever on results that
            no longer exist.
            """
            nonlocal _nc, js, psub, in_flight, pending_results, pending_expected
            nonlocal pending_publish_time, tid_to_instance_id, tid_to_dedup_key
            nonlocal pending_task_payload
            backoff = 1.0
            attempt = 0
            had_pending = in_flight > 0 or bool(pending_results)
            had_results_consumer_before = True
            while self.running:
                attempt += 1
                try:
                    # Probe whether the results consumer survived. If it did,
                    # any in-flight tasks can still be acknowledged; if not,
                    # we must reset state and let the data source replay.
                    try:
                        if _nc is not None and not _nc.is_closed:
                            from nats.js.errors import NotFoundError as _NotFoundError
                            try:
                                await js.consumer_info(
                                    cfg["results_stream"], cfg["results_consumer"],
                                )
                                had_results_consumer_before = True
                            except _NotFoundError:
                                had_results_consumer_before = False
                            except Exception:
                                had_results_consumer_before = False
                    except Exception:
                        had_results_consumer_before = False

                    _nc, js, psub = await _ensure_nats(
                        cfg, existing_nc=_nc, is_reconnect=True,
                    )

                    # If the durable consumer was recreated from scratch, the
                    # in-flight results are lost — re-publish the retained
                    # in-flight tasks so the gym workers re-run them (falling
                    # back to a reset + data_source replay when no payloads were
                    # retained). If the consumer was preserved, keep state so we
                    # can still ack incoming results.
                    if not had_results_consumer_before and had_pending:
                        if pending_task_payload:
                            # JetStream lost its state (stream and consumer
                            # recreated). Any results the gym workers published
                            # during the outage are gone and the recreated tasks
                            # stream is empty. Do NOT just reset and hope the
                            # data source replays — that dropped 2 of 64 groups
                            # in glm53 r6. Re-publish every retained in-flight
                            # task to the recreated tasks stream so the gym
                            # workers re-run it under the same tid; the
                            # pending_expected entries stay valid so the results
                            # still match.
                            replayed = 0
                            dropped = 0
                            for tid, (subject, data) in list(
                                pending_task_payload.items()
                            ):
                                try:
                                    await js.publish(subject, data)
                                except Exception:
                                    # Could not re-publish — drop this task and
                                    # let the data source replay the un-consumed
                                    # instance instead of hanging on it forever.
                                    pending_expected.pop(tid, None)
                                    pending_results.pop(tid, None)
                                    pending_publish_time.pop(tid, None)
                                    pending_task_payload.pop(tid, None)
                                    tid_to_instance_id.pop(tid, None)
                                    tid_to_dedup_key.pop(tid, None)
                                    dropped += 1
                                    continue
                                # Reset the deadline: the task starts over now.
                                pending_publish_time[tid] = time.time()
                                replayed += 1
                            in_flight = len(pending_expected)
                            logger.warning(
                                "NATS results consumer was lost — re-published "
                                "%d in-flight tasks to the recreated tasks "
                                "stream (dropped %d that failed re-publish; "
                                "in_flight now %d).",
                                replayed, dropped, in_flight,
                            )
                        else:
                            # No retained payloads to replay (e.g. resume from a
                            # checkpoint with no in-flight payloads) — reset and
                            # let the data source replay un-consumed instances.
                            logger.warning(
                                "NATS results consumer was lost and no in-flight "
                                "payloads were retained — resetting %d in-flight "
                                "tasks and %d pending results.",
                                in_flight, len(pending_results),
                            )
                            in_flight = 0
                            pending_results = {}
                            pending_expected = {}
                            pending_publish_time = {}
                            pending_task_payload = {}
                            tid_to_instance_id = {}
                            tid_to_dedup_key = {}
                    else:
                        logger.info(
                            "NATS reconnected — preserved %d in-flight tasks "
                            "and %d pending results.",
                            in_flight, len(pending_results),
                        )
                    return
                except Exception as exc:
                    logger.warning(
                        "NATS reconnect attempt %d failed: %s — retrying in %.1fs",
                        attempt, exc, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)

        logger.info(
            "NATS worker started: mode=%s, concurrency=%d, n_per_prompt=%d, "
            "inflight_multiplier=%d, max_in_flight=%d "
            "(batch mode: 1 msg per prompt, gym forks n_samples)",
            self.sample_mode, self.concurrency, self.n_per_prompt,
            self.inflight_multiplier, max_in_flight,
        )

        while self.running:
            try:
                # --- Publish tasks to keep in-flight pool full ---
                # Each published message = 1 prompt. The gym worker forks
                # n_samples_per_prompt workers and returns all trajectories
                # in a single result message.
                while in_flight < max_in_flight and self.running:
                    with self.data_source_lock:
                        groups = self.data_source.get_samples(1)

                    if not groups:
                        break

                    for group in groups:
                        sample = group[0]
                        meta = sample.metadata or {}
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except (json.JSONDecodeError, TypeError):
                                meta = {}

                        instance_id = str(meta.get("instance_id", sample.index or sample.group_index or 0))
                        # dedup_key is (instance_id, epoch) when the data source
                        # provides it; falls back to instance_id for backward
                        # compatibility / data sources that don't track epochs.
                        dedup_key = str(meta.get("dedup_key", instance_id))
                        if dedup_key in consumed_ids:
                            logger.debug("Skipping already-consumed instance %s (dedup=%s)", instance_id, dedup_key)
                            continue

                        gym_name = meta.get("gym_name", cfg["default_gym"])
                        task = sample_to_task(
                            sample,
                            rollout_id=group_counter,
                            group_index=group_counter,
                            n_samples=self.n_per_prompt,
                            gym_name=gym_name,
                            session=self.session,
                        )
                        # Make task_id unique per publish to avoid collisions
                        # when the data_source wraps epochs and re-emits the
                        # same instance_id while a prior copy is still in-flight.
                        # Suffix: ".g<counter>." — searchable in weave via
                        # contains ".g29." (won't match .g290. or .g292.).
                        # NOTE: cannot use "/" — task_id is used in filesystem paths.
                        raw_id = task.get("id", "unknown")
                        tid = f"{raw_id}.g{group_counter}."
                        task["id"] = tid
                        data = serialize_task(task)
                        if gym_name:
                            subject = f"{cfg['tasks_subject_prefix']}.{gym_name}"
                        else:
                            subject = cfg["tasks_subject_prefix"]
                        try:
                            await js.publish(subject, data)
                        except Exception as pub_exc:
                            if _is_nats_reconnect_error(pub_exc):
                                await _reconnect_nats()
                                break
                            raise
                        in_flight += 1
                        pending_expected[tid] = 1
                        pending_publish_time[tid] = time.time()
                        pending_task_payload[tid] = (subject, data)
                        tid_to_instance_id[tid] = raw_id
                        tid_to_dedup_key[tid] = dedup_key
                        if self.timing_tracker is not None and gym_name:
                            self.timing_tracker.record_publish(
                                tid, gym_name, time.time(),
                            )

                        group_counter += 1

                # --- Collect results from NATS ---
                try:
                    msgs = await psub.fetch(batch=min(in_flight, 16) or 1, timeout=5)
                except TimeoutError:
                    msgs = []
                except Exception as exc:
                    if _is_nats_reconnect_error(exc):
                        await _reconnect_nats()
                        continue
                    logger.error("Error fetching NATS results: %s", exc, exc_info=True)
                    await asyncio.sleep(2)
                    continue

                if msgs:
                    logger.info(
                        "Fetched %d result messages (in_flight=%d)",
                        len(msgs), in_flight,
                    )

                for msg in msgs:
                    try:
                        result = parse_result(msg.data)
                        await msg.ack()
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        logger.error("Bad message on results stream: %s", exc)
                        await msg.ack()
                        in_flight = max(0, in_flight - 1)
                        continue

                    tid = result.get("task_id", "unknown")
                    result_session = result.get("session")

                    # Verify gym_name survived NATS transit (paired with
                    # AREnABase _build_result log on the publish side).
                    # If pod log shows result.gym_name='foo' but this log
                    # shows result.gym_name='', the loss is in NATS / the
                    # JSON encoding path. CR-275800017's _task_gym fallback
                    # is only useful if the value is dropped between these
                    # two log points.
                    logger.info(
                        "Received result: task_id=%s result.gym_name=%r "
                        "has_gym_name_key=%s status=%s subject=%s session=%s",
                        tid, result.get("gym_name"),
                        "gym_name" in result, result.get("status"),
                        getattr(msg, "subject", "(unknown)"),
                        result_session,
                    )

                    # Drop results from a different trainer session — these
                    # come from a publish that the *previous* incarnation of
                    # this trainer made (e.g., before a checkpoint rollback).
                    # Without this filter, a stale in-flight result whose
                    # task_id we re-publish post-rollback would be silently
                    # accepted as the new attempt's result. result_session is
                    # None on results from older gym builds that don't echo
                    # the session — accept those for backwards compatibility
                    # (they predate this filter; nothing to compare against).
                    if (
                        result_session is not None
                        and result_session != self.session
                    ):
                        logger.info(
                            "Dropping result from prior session: task_id=%s "
                            "result_session=%s self.session=%s",
                            tid, result_session, self.session,
                        )
                        continue

                    # Drop results for task_ids we're not tracking — these are
                    # stale results from before a NATS reconnect, or from a
                    # rollout we already moved past. With session-filtering
                    # above, this is rare but still useful as a backstop for
                    # within-session redeliveries that landed late.
                    if tid not in pending_expected:
                        logger.info(
                            "Dropping stale result for unknown task_id %s "
                            "(likely from before NATS reconnect)", tid,
                        )
                        continue

                    in_flight = max(0, in_flight - 1)
                    pending_results.setdefault(tid, []).append(result)

                    logger.info(
                        "Result for %s: status=%s, accumulated=%d/%d",
                        tid, result.get("status"),
                        len(pending_results[tid]),
                        pending_expected[tid],
                    )

                    if self.mixture_controller:
                        gym_name = result.get("gym_name", "unknown")
                        self.mixture_controller.record_completion(gym_name)
                    if self.timing_tracker is not None:
                        self.timing_tracker.record_completion(tid, time.time())

                    # Check if we have all results for this task_id
                    expected = pending_expected[tid]
                    if len(pending_results[tid]) >= expected:
                        task_results = pending_results.pop(tid)
                        pending_expected.pop(tid, None)
                        pending_publish_time.pop(tid, None)
                        pending_task_payload.pop(tid, None)
                        instance_id = tid_to_instance_id.pop(tid, tid)
                        dedup_key = tid_to_dedup_key.pop(tid, instance_id)
                        self._process_group(tid, task_results, instance_id=instance_id, dedup_key=dedup_key)

                if self.mixture_controller:
                    new_weights = self.mixture_controller.maybe_adjust()
                    if new_weights is not None:
                        with self.data_source_lock:
                            if hasattr(self.data_source, "update_weights"):
                                self.data_source.update_weights(new_weights)

                if self.gym_autoscaler:
                    await self.gym_autoscaler.maybe_scale(js)

                # --- DLQ sweep: drop tasks whose retry budget is exhausted.
                # NATS gives up on a task after `max_deliver=3` attempts of
                # `ack_wait=3600s` each. After that the message is silently
                # discarded and no result will ever come back. Without this
                # sweep, the rollout would block forever waiting on
                # `pending_expected[tid]`. We treat such tasks as poison —
                # log a warning, decrement in_flight, and remove them from
                # bookkeeping. The corresponding instance is NOT marked
                # consumed in the data source, so it remains eligible for
                # re-emission on a future restart (where a different gym
                # build / model state might handle it). Within this run,
                # though, we don't auto-replay — that would just burn the
                # same compute again.
                now = time.time()
                if now - last_dlq_sweep >= dlq_sweep_interval_secs:
                    last_dlq_sweep = now
                    expired = [
                        t for t, ts in pending_publish_time.items()
                        if now - ts > task_deadline_secs
                    ]
                    if expired:
                        for tid in expired:
                            age = now - pending_publish_time.get(tid, now)
                            logger.warning(
                                "DLQ: task %s exceeded deadline (%.0fs > %.0fs) — "
                                "NATS retry budget likely exhausted; dropping from "
                                "in-flight tracking. Will re-emit on next worker "
                                "restart if not yet consumed.",
                                tid, age, task_deadline_secs,
                            )
                            pending_expected.pop(tid, None)
                            pending_results.pop(tid, None)
                            pending_publish_time.pop(tid, None)
                            pending_task_payload.pop(tid, None)
                            tid_to_instance_id.pop(tid, None)
                            tid_to_dedup_key.pop(tid, None)
                            in_flight = max(0, in_flight - 1)
                        logger.warning(
                            "DLQ swept %d expired tasks (in_flight now %d, pending_expected=%d).",
                            len(expired), in_flight, len(pending_expected),
                        )

                # Persist the timing tracker's state into the data source so
                # the next checkpoint carries it forward. This is cheap
                # (dict copy) and keeps the long-window baseline alive
                # across auto-restarts — throwing it away every 2 hours
                # would defeat the whole point of the trend signal.
                if self.timing_tracker is not None and hasattr(
                    self.data_source, "set_timing_tracker_state"
                ):
                    try:
                        self.data_source.set_timing_tracker_state(
                            self.timing_tracker.state_dict()
                        )
                    except Exception:
                        logger.debug(
                            "set_timing_tracker_state failed", exc_info=True,
                        )

                # Brief sleep if nothing to do
                if not msgs and in_flight >= max_in_flight:
                    await asyncio.sleep(1)

            except Exception as exc:
                if _is_nats_reconnect_error(exc):
                    await _reconnect_nats()
                else:
                    logger.error("Error in NATS worker loop: %s", exc, exc_info=True)
                    await asyncio.sleep(2)

        logger.info("NATS worker stopped")

    def _process_group(self, task_id: str, task_results: list[dict], *, instance_id: str | None = None, dedup_key: str | None = None):
        """Convert a completed group of results into Samples and put on output queue.

        Each NATS task result contains the trajectories from a fork pool of
        ``n_samples_per_prompt`` rollouts that the gym worker ran for one
        prompt. Group-level metrics (reward / length dispersion) and the
        OTel ``task.<id>`` parent span are emitted **gym-side** in the
        AREnABase dispatcher; this method just unpacks the trajectories
        into miles ``Sample`` objects.
        """
        # One episode per trajectory; an episode holds one Sample, or one per
        # compaction segment under --arena-train-segments all (ADR-0011).
        episodes: list[list[Sample]] = []
        n_failed = 0

        for result in task_results:
            # "truncated" envelopes stay usable: per-step stop_reason already
            # marks the affected samples TRUNCATED (remove_sample), so
            # dropping the whole envelope would discard the clean siblings.
            if result.get("status") not in SALVAGEABLE_RESULT_STATUSES:
                logger.warning("Task %s failed: %s", task_id, result.get("error"))
                n_failed += 1
                continue

            try:
                episodes.extend(
                    _result_to_episodes_full_trajectory(result, self._tokenizer, self.args)
                )
            except Exception as exc:
                logger.error("Task %s: conversion failed: %s", task_id, exc, exc_info=True)
                n_failed += 1

        if not episodes:
            logger.warning("Group %s: all %d tasks failed, dropping", task_id, n_failed)
            return

        if n_failed > 0:
            logger.warning(
                "Group %s: %d/%d failed, padding with sample copies",
                task_id, n_failed, len(task_results),
            )

        # Pad or trim to n_samples_per_prompt EPISODES with real sample copies.
        # The pad is a one-sample episode copied from the first episode's final
        # segment (the top-level rollout.json shape).
        while len(episodes) < self.n_per_prompt:
            src = episodes[0][-1]
            pad = Sample()
            pad.reward = 0.0
            pad.tokens = list(src.tokens)
            pad.loss_mask = [0] * src.response_length
            pad.response_length = src.response_length
            # rollout_log_probs MUST be set whenever the real samples carry it
            # (use_rollout_logprobs path): the actor gates the whole batch on
            # samples[0] and then slices EVERY sample, asserting
            # len(rollout_log_probs) == response_length. A pad left at the
            # Sample default (None) would hit `len(None)` in slice_log_prob_with_cp.
            # Mirror the source sample: emit a zero list of the same length when
            # it carries logprobs, else leave None (re-tokenize path, no logprobs).
            if src.rollout_log_probs is not None:
                pad.rollout_log_probs = [0.0] * pad.response_length
            pad.remove_sample = True
            # Tag pads as "failed" so the rollout-level failed-fraction metric
            # can count them. A pad fills a slot left by a gym-side failure: a
            # synthetic/killed trajectory skipped in _result_to_episodes, or an
            # entirely failed task result. (remove_sample alone can't be
            # attributed — truncation/bad-logprob drops also set it.)
            pad.status = Sample.Status.FAILED
            # Match the metadata shape of real samples (task_id, gym_name,
            # segment / n_segments) so downstream handling (metric attribution,
            # OTel, dedup injection below) treats pads uniformly rather than
            # special-casing missing keys. gym_name is copied from a real
            # sibling (episodes is non-empty here — the all-failed case
            # returned above). A pad is a one-sample episode (ADR-0011).
            pad.metadata = {
                "mode": "failed",
                "task_id": task_id,
                "gym_name": (src.metadata or {}).get("gym_name", "unknown"),
                "segment": 0,
                "n_segments": 1,
            }
            episodes.append([pad])
        if len(episodes) > self.n_per_prompt:
            episodes = episodes[:self.n_per_prompt]

        # Inject raw instance_id (stable, OTel-friendly) and dedup_key
        # (instance_id+epoch, used by the resume guard) into sample metadata.
        if instance_id:
            for episode in episodes:
                for s in episode:
                    meta = s.metadata if isinstance(s.metadata, dict) else {}
                    meta["instance_id"] = instance_id
                    if dedup_key:
                        meta["dedup_key"] = dedup_key
                    s.metadata = meta

        # Stamp miles Sample identity across all siblings of this prompt-group:
        # a shared ``group_index`` (prompt identity) plus a unique per-sample
        # ``index``. ``group_index`` keys the GRPO reward-normalization
        # segments (miles _reward_group_segments) explicitly instead of relying
        # on the contiguous n_samples_per_prompt layout; ``index`` makes
        # ``sample_indices`` / ``rollout_ids`` well-defined ints (miles packs
        # them into int64 arrays, so None would crash) and makes each
        # trajectory its own loss-aggregation unit (``rollout_mask_sums`` —
        # masked/removed samples contribute 0 to the loss).
        #
        # PORT NOTE (drift from the slime 0.3.0 overlay): the original stamped
        # a shared ``Sample.group_id`` per group so slime's
        # _convert_samples_to_train_data keyed per-group loss denominators
        # (group_mask_sums) on it. miles has no ``group_id``; its
        # ``Sample.rollout_id`` is a *different* concept — compact siblings of
        # ONE rollout execution, which must share a single reward and are
        # counted once by the trainer. Stamping the shared GROUP id onto
        # ``rollout_id`` would make miles' reward normalization
        # (_normalize_rewards_by_rollout) demand one reward for the whole
        # GRPO group (ValueError on any within-group reward variance) and
        # would disable batch trimming, so in ``final`` mode ``rollout_id`` is
        # deliberately left None and each trajectory is represented the way
        # standard miles GRPO fan-out samples are (ADR-0003).
        #
        # ``--arena-train-segments all`` (ADR-0011) stamps the EPISODE id
        # ``base = gid * n + e`` on ``rollout_id`` of every segment of episode
        # ``e`` — exactly miles' documented meaning (siblings of one rollout
        # execution, one reward) — and spreads ``index`` by
        # ``_SEGMENT_INDEX_STRIDE`` so it stays unique. ``k == 0`` reproduces
        # today's ``index``; the batch is then compact (no sample-count trim).
        gid = self._output_group_counter
        self._output_group_counter += 1
        stamp_rollout_id = _train_segments_mode(self.args) == "all"
        for e, episode in enumerate(episodes):
            base = gid * self.n_per_prompt + e
            for k, s in enumerate(episode):
                s.group_index = gid
                s.index = base + k * _SEGMENT_INDEX_STRIDE
                if stamp_rollout_id:
                    s.rollout_id = base

        self.output_queue.put([s for episode in episodes for s in episode])

    def _worker_thread_func(self):
        asyncio.run(self._worker_loop())

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker_thread_func, daemon=True)
            self.worker_thread.start()
            logger.info("Started NATS rollout worker thread")

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
        logger.info("Stopped NATS rollout worker thread")

    def get_completed_groups(self) -> list[list[Sample]]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


def _pad_rows_to_dp_alignment(data: list[list[Sample]], args: Any) -> int:
    """Pad the ``all``-mode batch so ``build_dp_schedule`` can align it.

    ``build_dp_schedule`` needs the micro-batch count of a step to be a multiple
    of ``dp_size * mb_group``. The dynamic path grows the count by splitting
    multi-sample micro-batches (``expand_bins_by_splitting``), but a row that
    alone exceeds ``max_tokens_per_gpu`` is a singleton it cannot split. The
    static path never splits. ``generate_rollout`` delivers exactly
    ``global_batch_size`` episodes, but the row count is the sum of their
    segments. An odd row count on ``dp_size == 2`` therefore asserts on the
    rollout side: ``RolloutManager.generate`` -> ``split_train_data_by_dp`` ->
    ``split_train_data_by_dp_scheduled_raw`` -> ``build_dp_schedule``. Under
    ``delay_split_train_data_by_dp`` or ``indep_dp`` (``train_parallel_config``
    is ``{}``) the legacy ``split_train_data_by_dp_raw`` runs instead. There
    ``--balance-data`` asserts ``rows % dp_size == 0``
    (``get_seqlen_balanced_partitions(equal_size=True)``); ``unit`` is a
    multiple of ``dp_size``, so the pad covers that path too.

    Each pad is a zero-loss sibling segment of the shortest kept row: same
    ``tokens``, ``reward``, ``group_index`` and ``rollout_id``, all-zero
    ``loss_mask``. ``_normalize_rewards_by_rollout`` sees one more row with the
    episode reward and ``_compute_rollout_mask_sums`` adds zero, so the
    episode's advantage and loss weight are unchanged. ``remove_sample`` stays
    False so the per-episode ``removed_frac`` does not flag the episode; the
    pad is visible as ``metadata["mode"] == "dp_pad"``.

    Args:
        data: The post-filter groups. Pads are appended to the source's group.
        args: Trainer args; reads the actor parallel sizes and the batch mode.

    Returns:
        The number of rows added (0 when the batch is already aligned).
    """
    # ponytail: mirrors build_dp_schedule's align_to. vpp_size reads the arg that
    # Megatron validate_args sets; mb_group mirrors parallel.py _compute_vpp_fields
    # (pipeline_model_parallel_size when vpp > 1). tp/pp default to 1 because
    # FSDPArgs defines context_parallel_size only, and FSDP dp is world // cp.
    # The upgrade path is an upstream fix in build_dp_schedule (pad or let a
    # rank take zero micro-batches); this helper then becomes a no-op and can go.
    rows = sum(len(g) for g in data)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    pp_size = getattr(args, "pipeline_model_parallel_size", 1)
    model_parallel = tp_size * pp_size * args.context_parallel_size
    dp_size = (args.actor_num_nodes * args.actor_num_gpus_per_node) // model_parallel
    vpp_size = getattr(args, "virtual_pipeline_model_parallel_size", None) or 1
    align_to = dp_size * (pp_size if vpp_size > 1 else 1)
    unit = align_to if args.use_dynamic_batch_size else align_to * args.micro_batch_size
    pads = (-rows) % unit
    if pads == 0:
        return 0

    flat = [s for g in data for s in g]
    src = min([s for s in flat if not s.remove_sample] or flat, key=lambda s: len(s.tokens))
    group = next(g for g in data if any(s is src for s in g))
    episode = [s for s in group if _episode_key(s) == _episode_key(src)]
    base = min(s.index for s in episode)
    for k in range(pads):
        pad = Sample()
        pad.tokens = list(src.tokens)
        pad.response_length = src.response_length
        pad.loss_mask = [0] * src.response_length
        # Mirror the source: the actor slices EVERY sample's rollout_log_probs
        # when samples[0] carries them (see the failed-pad comment in _process_group).
        if src.rollout_log_probs is not None:
            pad.rollout_log_probs = [0.0] * src.response_length
        pad.reward = src.reward
        pad.group_index = src.group_index
        pad.rollout_id = src.rollout_id
        pad.index = base + (len(episode) + k) * _SEGMENT_INDEX_STRIDE
        # COMPLETED, never the source status: a zero-loss row trains nothing, and a
        # copied TRUNCATED inflates rollout/truncated_ratio and train_data["truncated"].
        pad.status = Sample.Status.COMPLETED
        pad.weight_versions = list(src.weight_versions)
        pad.remove_sample = False
        pad.metadata = {**(src.metadata or {}), "mode": "dp_pad", "segment": len(episode) + k}
        group.append(pad)
    logger.warning(
        "DP alignment: %d rows + %d zero-loss pad(s) -> multiple of %d; "
        "build_dp_schedule cannot split singleton micro-batches (rows over max_tokens_per_gpu) "
        "and the static path never splits",
        rows,
        pads,
        unit,
    )
    return pads


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------

def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """NATS rollout function entry point.

    Uses a global background worker that continuously publishes tasks to
    NATS and collects results. Returns as soon as GBS-worth of groups are
    ready (global_batch_size / n_samples_per_prompt), while the background
    worker keeps collecting into the buffer for subsequent calls.

    The background worker publishes up to rollout_batch_size prompts
    in-flight to keep gym workers saturated. This decouples collection
    throughput from training step consumption.
    """
    if evaluation:
        raise NotImplementedError("Evaluation mode not yet supported for NATS rollout")

    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_source)
    target_groups = args.global_batch_size // args.n_samples_per_prompt

    # --- Dynamic sampling (DAPO) -------------------------------------------
    # Drop groups with no reward variance (all-pass or all-fail): their GRPO
    # advantage is identically 0, so they cost rollout compute but produce no
    # gradient. The filter is loaded from --dynamic-sampling-filter-path (e.g.
    # miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std).
    # Unlike sglang_rollout (which over-samples synchronously), here we drain
    # extra groups from the already-over-provisioned output_queue until we have
    # target_groups survivors. ``max_examined`` bounds how many groups we'll
    # inspect so a batch where almost everything is zero-variance can't stall
    # the step forever — we accept whatever we have once the cap is hit.
    from miles.rollout.filter_hub.base_types import call_dynamic_filter

    dyn_filter = None
    filter_path = getattr(args, "dynamic_sampling_filter_path", None)
    if filter_path:
        from miles.utils.misc import load_function
        dyn_filter = load_function(filter_path)
        logger.info("NATS dynamic sampling enabled: filter=%s", filter_path)

    # Budget for how many groups the filter will inspect before giving up.
    # E.g. with target_groups=16 and examine_mult=4.0, max_examined=64:
    # the filter looks at up to 64 groups to fill 16 slots. If 50% are
    # zero-variance, ~32 examined fills the batch. The 4x cap gives
    # headroom. Once the cap is hit, remaining slots are filled unfiltered
    # to prevent the step from stalling when most tasks produce no signal.
    examine_mult = float(getattr(args, "dynamic_sampling_max_examine_mult", 4.0) or 4.0)
    max_examined = max(target_groups, int(target_groups * examine_mult))

    data: list[list[Sample]] = []
    # Every group pulled from the queue, BEFORE dynamic-sampling filtering.
    # The reward-collapse metrics are computed over this pre-filter
    # population (mirroring sglang_rollout's ``all_data``); ``data`` holds only
    # the post-filter groups that actually train. When no filter is active
    # ``all_data == data``.
    all_data: list[list[Sample]] = []
    examined = 0
    dropped = 0
    start_time = time.time()
    last_log = start_time
    do_print = True

    queue_depth_at_start = worker.get_queue_size()
    logger.info(
        "Rollout %d: collecting %d groups (GBS=%d, n_samples=%d, queue=%d)",
        rollout_id, target_groups, args.global_batch_size,
        args.n_samples_per_prompt, queue_depth_at_start,
    )

    while len(data) < target_groups:
        remaining = target_groups - len(data)
        for _ in range(remaining):
            try:
                group = worker.output_queue.get_nowait()
            except queue.Empty:
                break

            # Record the group pre-filter so collapse metrics see the true
            # population (a zero-variance group is exactly what we want to
            # measure, and exactly what the filter would otherwise hide).
            all_data.append(group)

            if dyn_filter is not None and examined < max_examined:
                examined += 1
                out = call_dynamic_filter(dyn_filter, args, group)
                if not out.keep:
                    dropped += 1
                    if dropped <= 5 or dropped % 20 == 0:
                        logger.info(
                            "dyn-sampling drop #%d (reason=%s, examined=%d, kept=%d/%d)",
                            dropped, out.reason, examined, len(data), target_groups,
                        )
                    continue

            data.append(group)

            if do_print:
                logger.info(
                    "First group: %d samples, reward=%s, tokens=%d",
                    len(group), group[0].reward, len(group[0].tokens),
                )
                do_print = False

        now = time.time()
        if now - last_log > 30:
            logger.info(
                "Waiting for results: %d/%d groups collected (%.0fs elapsed, queue=%d)",
                len(data), target_groups, now - start_time, worker.get_queue_size(),
            )
            last_log = now
            # Refresh the eval-metrics heartbeat from the rollout wait
            # loop. Without this, ``touch_trainer_alive`` only fires after
            # a successful train iter (in ``log_perf_data_raw``); a long
            # rollout (>10 min, the eval coordinator's stale threshold)
            # then looks like trainer death and the next eval coordinator
            # opens its own wandb session, racing the trainer on _step.
            try:
                from miles_plugins.arena.logging_extensions import touch_trainer_alive
                touch_trainer_alive(args)
            except Exception:
                pass

        if len(data) < target_groups:
            if not worker.worker_thread.is_alive():
                logger.error(
                    "NATS worker thread died during collection (%d/%d groups). "
                    "Returning partial batch to avoid infinite hang.",
                    len(data), target_groups,
                )
                break
            time.sleep(0.5)

    duration = time.time() - start_time

    if dyn_filter is not None:
        logger.info(
            "Dynamic sampling: kept=%d/%d, dropped=%d (zero-variance), examined=%d, cap=%d%s",
            len(data), target_groups, dropped, examined, max_examined,
            " [HIT CAP — accepted unfiltered tail]" if examined >= max_examined and len(data) < target_groups else "",
        )

    # --arena-train-segments all: every sample carries a rollout_id, so miles
    # runs the batch in compact mode and postprocess_rollout_data does NOT trim
    # by sample count; the legacy get_data_iterator would then count samples,
    # not episodes. A partial batch (worker death) must fail here, loudly, like
    # today's "Not enough samples" (ADR-0011).
    if worker._train_segments == "all":
        n_episodes = len(_episode_representatives(data))
        if n_episodes < args.global_batch_size:
            raise RuntimeError(
                f"arena: {n_episodes} episodes < global_batch_size {args.global_batch_size}; "
                "compact mode does not trim and the legacy iterator would count samples"
            )
        # The row count is the sum of segments, so its DP alignment is
        # data-dependent; pad here rather than assert in build_dp_schedule.
        _pad_rows_to_dp_alignment(data, args)

    # Every count below is per EPISODE (ADR-0011): a multi-segment episode under
    # --arena-train-segments all is one unit, so the numbers match ``final``.
    tm = _batch_telemetry(data, all_data)
    total_episodes = int(tm["total_episodes"])
    avg_reward = tm["avg_reward"]
    logger.info(
        "Reward distribution: nonzero_samples=%d/%d, nonzero_groups=%d/%d, max_reward=%.3f",
        tm["nonzero_count"], total_episodes, tm["nonzero_groups"], len(data), tm["max_reward"],
    )
    failed_frac = tm["failed_frac"]
    removed_frac = tm["removed_frac"]
    truncated_ratio = tm["truncated_ratio"]
    logger.info(
        "Failed-sample distribution: failed=%d/%d (%.3f), removed_total=%d/%d (%.3f)",
        tm["failed_count"], total_episodes, failed_frac,
        tm["removed_count"], total_episodes, removed_frac,
    )
    # Per-reason breakdown of removed_total so a rollout log shows WHY
    # training signal was lost without a post-hoc investigation. Reasons are
    # stamped into Sample.metadata["removal_reason"] by
    # _result_to_episodes_full_trajectory: the degenerate agent stop
    # ("timeout", "context_error", ...), "length" (per-turn clip),
    # "context_overflow", "bad_logprobs"/"no_logprobs"; gym-side pads carry
    # mode="failed". kept_timeout counts the (non-removed) samples salvaged
    # by --arena-keep-timeout-trajectories; timeout=N under the flag then
    # means "timeout AND another defect" (see precedence note there).
    # Counted per EPISODE (ADR-0011) so the breakdown sums to removed_total
    # above: one sample per episode, the removed segment when there is one,
    # because the first segment can train while a later one is dropped.
    removal_reasons, kept_timeout_count, kept_context_error_count = _removal_reason_counts(
        [[_removal_representative(e) for e in _episodes(data)]]
    )
    logger.info(
        "Removal reasons: %s (kept_timeout=%d, kept_context_error=%d)",
        ", ".join(
            f"{k}={v}"
            for k, v in sorted(removal_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        ) or "none",
        kept_timeout_count,
        kept_context_error_count,
    )
    # -----------------------------------------------------------------------

    gym_counts: dict[str, int] = {}
    for group in data:
        for sample in group:
            meta = sample.metadata or {}
            gn = meta.get("gym_name", "unknown")
            gym_counts[gn] = gym_counts.get(gn, 0) + 1
            break
    gym_str = ", ".join(f"{g}={c}" for g, c in sorted(gym_counts.items()))

    queue_depth_at_end = worker.get_queue_size()
    logger.info(
        "Rollout %d complete: %d groups (%s), %d samples, avg_reward=%.3f in %.1fs (queue=%d)",
        rollout_id, len(data), gym_str, total_episodes, avg_reward, duration,
        queue_depth_at_end,
    )

    # Surface NATS rollout queue depth to W&B so we can see in a graph
    # whether the publisher is keeping ahead of the trainer.
    #   - queue_depth_at_start: groups buffered when trainer asked for the batch
    #   - queue_depth_at_end: groups buffered after we took target_groups out
    #     (= safety margin for the next training step)
    #   - groups_collected_during_wait: how many groups arrived from gym workers
    #     while this rollout was assembling (i.e. delta the trainer wait window
    #     produced). queue_depth_at_end + target_groups - queue_depth_at_start.
    # Record consumed instance_ids for resume tracking.
    # On checkpoint, these are persisted via data_source.save() so that on
    # restart the worker can skip already-trained prompts.
    # Prefer instance_id (raw prompt id) over task_id (unique per-publish id)
    # so the resume skip logic matches what data_source emits.
    consumed_ids = []
    consumed_task_ids = []
    consumed_instance_ids = []
    consumed_gym_names = []
    for group in data:
        for sample in group:
            meta = sample.metadata or {}
            # Resume guard checks dedup_key (instance_id+epoch). Fall back to
            # instance_id, then task_id, for backward compatibility.
            key = meta.get("dedup_key") or meta.get("instance_id") or meta.get("task_id", "")
            tid = meta.get("task_id", "")
            iid = meta.get("instance_id", "")
            gn = meta.get("gym_name", "")
            if key:
                consumed_ids.append(str(key))
                consumed_task_ids.append(str(tid))
                consumed_instance_ids.append(str(iid))
                consumed_gym_names.append(str(gn))
                break
    logger.info(
        "rollout_tasks: rollout=%d task_ids=%s",
        rollout_id, consumed_task_ids or consumed_ids,
    )
    if hasattr(data_source, "record_consumed_samples"):
        data_source.record_consumed_samples(rollout_id, consumed_ids)

    if getattr(args, "use_wandb", False):
        try:
            from miles.utils.metric_utils import compute_rollout_step
            from miles.utils.tracking_utils.tracking import log as _wandb_log
            step = compute_rollout_step(args, rollout_id)
            groups_collected_during_wait = (
                queue_depth_at_end + target_groups - queue_depth_at_start
            )
            metrics = {
                "rollout/queue_depth_at_start": queue_depth_at_start,
                "rollout/queue_depth_at_end": queue_depth_at_end,
                "rollout/groups_collected_during_wait": groups_collected_during_wait,
                "rollout/step": step,
                # reward-collapse telemetry (per episode, ADR-0011)
                "rollout/reward_nonzero_frac": tm["reward_nonzero_frac"],
                "rollout/zero_reward_groups_frac": tm["zero_reward_groups_frac"],
                "rollout/group_std": tm["mean_group_std"],
                "rollout/reward_p25": tm["reward_p25"],
                "rollout/reward_p50": tm["reward_p50"],
                "rollout/reward_p75": tm["reward_p75"],
                "rollout/reward_p90": tm["reward_p90"],
                "rollout/avg_response_length": tm["avg_response_length"],
                # failed/dropped-sample telemetry
                "rollout/failed_frac": failed_frac,
                "rollout/removed_sample_frac": removed_frac,
                # Pre-filter population (all_data); miles-native log_rollout_data
                # logs the post-filter rollout/truncated_ratio at the same step.
                "rollout/truncated_ratio_prefilter": truncated_ratio,
                # Segment counts: miles log_rollout_data logs
                # rollout/num_training_samples and rollout/episode_raw_reward
                # for this batch (ADR-0011), so nothing to add here.
                # dynamic-sampling (DAPO) telemetry
                "rollout/dyn_sampling_dropped": dropped,
                "rollout/dyn_sampling_drop_frac": (
                    dropped / examined if examined > 0 else 0.0
                ),
            }

            # Aggregate gym-side group_metrics (num_turns, completion_length,
            # per-group reward stats) across the batch.
            from miles_plugins.arena.rollout_metrics import (
                compute_group_metrics_from_samples,
                compute_off_policy_metrics,
            )

            # One representative per episode: group_metrics sits on the first
            # sample of a group (always a representative), and the off-policy /
            # binary-reward metrics must count an episode once (ADR-0011).
            all_samples = _episode_representatives(all_data)
            gm = compute_group_metrics_from_samples(all_samples)
            if gm:
                for k, v in gm.items():
                    metrics[f"rollout/group_metrics/{k}"] = v

            # Off-policy staleness (rollout/off_policy_round/*): how far the
            # trainer's weights advanced past the weights that generated each
            # sample. Reads s.weight_versions, which miles tags natively via the
            # SGLang weight-version handshake (types.py update_from_meta_info),
            # so no manager-side propagation is needed. reference_step ==
            # rollout_id (one generate_rollout == one trainer step in
            # nats_rollout).
            off_policy = compute_off_policy_metrics(args, all_samples, rollout_id)
            if off_policy:
                for k, v in off_policy.items():
                    metrics[f"rollout/{k}"] = v

            # Binary reward (rollout/binary_reward): fraction of samples with a
            # reward >= 1.0. Only meaningful when the run binarizes rewards, so
            # gate on the post-processor path (mirrors gym-evals rollout.py).
            if getattr(args, "custom_reward_post_process_path", None) and "binary" in args.custom_reward_post_process_path:
                raw_rewards = [s.get_reward_value(args) for s in all_samples]
                if raw_rewards:
                    metrics["rollout/binary_reward"] = sum(
                        1.0 if r >= 1.0 else 0.0 for r in raw_rewards
                    ) / len(raw_rewards)

            _wandb_log(args, metrics, step_key="rollout/step")

            # Step <-> task mapping table: lets us go from a train step in the
            # W&B UI to the exact task_ids consumed there, then jump into the
            # trace search filtered by `task.<task_id>` to inspect the
            # trajectories that produced this gradient update.
            #
            # In nats_rollout, generate_rollout collects exactly
            # global_batch_size // n_samples_per_prompt groups = one trainer
            # batch per call. So one rollout_id == one trainer step. We use
            # rollout_id directly for train_step instead of
            # compute_rollout_step (which is calibrated for standard rollouts
            # where each generate_rollout returns multiple batches and
            # therefore advances the trainer step by several units).
            try:
                import wandb
                rows = [
                    [rollout_id, rollout_id, tid, iid, gn]
                    for tid, iid, gn in zip(
                        consumed_task_ids,
                        consumed_instance_ids,
                        consumed_gym_names,
                    )
                ]
                table = wandb.Table(
                    columns=["train_step", "rollout_id", "task_id", "instance_id", "gym_name"],
                    data=rows,
                )
                _wandb_log(
                    args,
                    {"rollout/task_assignments": table},
                    step_key="rollout/step",
                )
            except Exception as exc:
                logger.warning("Failed to log task_assignments table to W&B: %s", exc)
        except Exception as exc:
            logger.warning("Failed to log queue metrics to W&B: %s", exc)

    return data


def _add_arena_arguments(parser):
    """Register the arena-only CLI knobs this module reads.

    miles' parser is strict (unknown flags are a hard startup failure, unlike
    the vendored slime 0.3.0 megatron parser which silently dropped them), and
    parse_args auto-invokes ``add_arguments`` on the object resolved from
    ``--rollout-function-path`` (miles.utils.arguments,
    add_user_provided_function_arguments). Defaults here are IDENTICAL to the
    getattr fallbacks at the read sites; the getattr-with-default + env-var
    reads stay in place as belt-and-suspenders for invocations where this
    hook never ran (e.g. MILES_USE_LEGACY_ROLLOUT_V1). Boolean knobs are bare
    flags because the hydra converter only emits a flag when the YAML value
    is true; ``--gym-autoscale-auto-tune`` defaults True (matching its getattr
    fallback), so the flag itself is a no-op kept for YAML compatibility.

    Knobs read by other arena modules (data source, eval path, coordinator)
    are NOT registered here — only what nats_rollout itself consumes.
    """
    group = parser.add_argument_group(title="arena nats rollout")
    group.add_argument(
        "--arena-sample-mode",
        type=str,
        default=None,
        help="Arena sample conversion mode; only 'full_trajectory' is "
        "supported. Falls back to the ARENA_SAMPLE_MODE env var.",
    )
    group.add_argument(
        "--arena-train-segments",
        type=str,
        choices=("final", "all"),
        default="final",
        help="final = train the last step only (today); all = one Sample per "
        "compaction segment sharing one rollout_id (ADR-0011). 'all' expands "
        "a step list only when a step carries the gym's segment_end marker.",
    )
    group.add_argument(
        "--arena-inflight-multiplier",
        type=int,
        default=2,
        help="Publisher in-flight cap as a multiple of rollout_batch_size "
        "(max_in_flight = multiplier x rollout_batch_size). 2 keeps the "
        "r10-lineage oversubscription; raise it when SGLang engines sit "
        "under-fed with an empty queue.",
    )
    group.add_argument(
        "--dynamic-sampling-max-examine-mult",
        type=float,
        default=4.0,
        help="Cap on groups the DAPO dynamic-sampling filter may examine, "
        "as a multiple of target_groups.",
    )
    group.add_argument(
        "--gym-mixture-targets",
        type=str,
        default=None,
        help="JSON dict of per-gym target sampling ratios; enables the "
        "MixtureController feedback loop.",
    )
    group.add_argument("--mixture-adjustment-interval", type=float, default=60.0)
    group.add_argument("--mixture-smoothing", type=float, default=0.3)
    group.add_argument(
        "--k8s-namespace",
        type=str,
        default=None,
        help="K8s namespace for autoscaler Deployment patches; falls back "
        "to the NAMESPACE env var, then 'default'.",
    )
    group.add_argument(
        "--arena-mask-clipped-final-turn",
        action="store_true",
        default=False,
        help="When a trajectory's final generate hit the per-turn token cap "
        "(the gym stamps stop_reason=length only for the final turn), zero "
        "only that clipped final turn's loss-mask tokens and keep the "
        "earlier turns trainable, instead of removing the whole sample "
        "(the r5-lineage default).",
    )
    group.add_argument(
        "--arena-keep-timeout-trajectories",
        action="store_true",
        default=False,
        help="Keep token-level (GenerateClient) trajectories whose agent loop "
        "stopped with agent_stop_reason=timeout (the Harbor agent hit its "
        "per-task wall-clock deadline) as training samples when that is the "
        "only defect: loss_mask untouched, remove_sample stays False, status "
        "stays TRUNCATED. Timeout trajectories that also have a clipped "
        "final turn, a hard context overflow or zero-filled logprobs are "
        "still removed (a clipped final turn can be salvaged by "
        "--arena-mask-clipped-final-turn). Default: remove them "
        "(the r5-lineage default).",
    )
    group.add_argument(
        "--arena-keep-context-error-trajectories",
        action="store_true",
        default=False,
        help="Keep token-level (GenerateClient) trajectories whose agent loop "
        "stopped with agent_stop_reason=context_error (the arena Terminus-2 "
        "agent ended the episode because the next turn's prompt exceeded the "
        "engine context window) as training samples when that is the only "
        "defect: loss_mask untouched, remove_sample stays False, status stays "
        "TRUNCATED. Same precedence rules as "
        "--arena-keep-timeout-trajectories. Default: remove them.",
    )
    group.add_argument("--gym-autoscale", action="store_true", default=False)
    group.add_argument(
        "--gym-autoscale-config",
        type=str,
        default=None,
        help="JSON dict {gym: {deploy_name, concurrency_per_pod, "
        "min/max/initial_replicas}}; falls back to GYM_AUTOSCALE_CONFIG env.",
    )
    group.add_argument("--gym-autoscale-auto-tune", action="store_true", default=True)
    group.add_argument("--gym-autoscale-profile", action="store_true", default=False)
    group.add_argument("--gym-autoscale-profile-only", action="store_true", default=False)
    group.add_argument("--gym-autoscale-dry-run", action="store_true", default=False)
    group.add_argument("--gym-autoscale-short-window", type=int, default=100)
    group.add_argument("--gym-autoscale-long-window", type=int, default=1000)
    group.add_argument("--gym-autoscale-interval", type=float, default=30.0)
    group.add_argument("--gym-autoscale-cooldown", type=float, default=180.0)
    group.add_argument("--gym-autoscale-warmup", type=float, default=1800.0)
    group.add_argument("--gym-autoscale-headroom", type=float, default=2.0)
    group.add_argument("--gym-autoscale-deficit-alpha", type=float, default=0.5)
    group.add_argument("--gym-autoscale-growth-threshold", type=float, default=1.3)
    group.add_argument("--gym-autoscale-growth-preemption", type=float, default=1.25)
    group.add_argument("--gym-autoscale-profile-duration", type=float, default=1800.0)
    group.add_argument("--gym-autoscale-min-samples", type=int, default=50)
    return parser


generate_rollout.add_arguments = _add_arena_arguments

atexit.register(stop_global_worker)
