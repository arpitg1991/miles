"""Every verified episode trains; only a broken token stream leaves the loss.

The gym masks a clipped or empty generate in ``loss_mask`` itself and its
verifier runs on every trajectory it ships, so ``agent_stop_reason`` and a
final ``stop_reason == "length"`` are telemetry. A sample is removed only for
a hard context overflow (``context_overflow``) or a rollout-logprob length
mismatch (``bad_logprobs``). The plugin never sets ``advantage_scale``.

Run: python -m pytest tests/fast/plugins/arena/test_stop_reason_policy.py -v
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import (
    _add_arena_arguments,
    _removal_reason_counts,
    _result_to_samples_full_trajectory,
    _stop_metrics,
)

# Prompt, turn A (2 toks), tool output, turn B (3 toks). Response window = from the first 1.
_MULTI_TURN = [0, 1, 1, 0, 1, 1, 1]
_RESPONSE = _MULTI_TURN[1:]


def _step(
    loss_mask: list[int] = _MULTI_TURN, stop_reason: str = "stop", log_probs: list[float] | None = None
) -> dict:
    # log_probs aligned 1:1 with token_ids keeps the rollout-logprob check green.
    return {
        "token_ids": [7] * len(loss_mask),
        "loss_mask": list(loss_mask),
        "log_probs": [-0.1] * len(loss_mask) if log_probs is None else log_probs,
        "stop_reason": stop_reason,
        "has_generate_tokens": True,
    }


def _result(step: dict, **traj: object) -> dict:
    return {"task_id": "t1", "gym_name": "g", "trajectories": [{"reward": 1.0, "steps": [step], **traj}]}


def _one(result: dict, index: int = 0, **args: object) -> Sample:
    (s,) = _result_to_samples_full_trajectory(result, tokenizer=None, args=SimpleNamespace(**args))
    # Episode identity for the telemetry helpers, as _process_group stamps it.
    s.index = index
    s.group_index = 0
    return s


# (a) every stop reason is kept, loss_mask untouched, status COMPLETED --------
@pytest.mark.parametrize(
    "reason", ["timeout", "context_error", "empty_response", "truncated", "max_budget", "completed", None]
)
def test_agent_stop_reason_is_telemetry(reason: str | None) -> None:
    traj = {"agent_stop_reason": reason} if reason else {}
    s = _one(_result(_step(), **traj))
    assert s.status == Sample.Status.COMPLETED
    assert s.remove_sample is False
    assert s.loss_mask == _RESPONSE
    assert len(s.rollout_log_probs) == s.response_length
    assert s.reward == 1.0
    assert "removal_reason" not in s.metadata
    assert s.metadata.get("agent_stop_reason") == reason


def test_clipped_final_generate_is_kept_with_the_gym_mask() -> None:
    # The gym already zeroed the clipped turn B; the trainer leaves the mask alone.
    masked = [0, 1, 1, 0, 0, 0, 0]
    s = _one(_result(_step(masked, stop_reason="length"), agent_stop_reason="truncated", truncated_turns=1))
    assert s.status == Sample.Status.COMPLETED
    assert s.remove_sample is False
    assert s.loss_mask == masked[1:]
    assert s.metadata["truncated_turns"] == 1


def test_legacy_caller_without_args_keeps_the_sample() -> None:
    (s,) = _result_to_samples_full_trajectory(
        _result(_step(stop_reason="length"), agent_stop_reason="timeout"), tokenizer=None, args=None
    )
    assert s.status == Sample.Status.COMPLETED
    assert s.remove_sample is False


# (b) a broken token stream is still removed ---------------------------------
def test_hard_overflow_is_removed_as_context_overflow() -> None:
    s = _one(_result(_step(), agent_stop_reason="completed"), rollout_max_context_len=4)
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert len(s.tokens) == 4
    assert s.metadata["removal_reason"] == "context_overflow"


def test_bad_logprobs_is_removed() -> None:
    s = _one(_result(_step(log_probs=[-0.1] * 3), agent_stop_reason="completed"))  # misaligned with tokens
    assert s.status == Sample.Status.ABORTED
    assert s.remove_sample is True
    assert s.rollout_log_probs == [0.0] * s.response_length
    assert s.metadata["removal_reason"] == "bad_logprobs"


# (c) stop telemetry from the trajectory counters -----------------------------
def test_stop_metrics_from_trajectory_counters() -> None:
    a = _one(_result(_step(), agent_stop_reason="Completed", truncated_turns=2, masked_output_tokens=40), index=0)
    b = _one(_result(_step(), agent_stop_reason="timeout"), index=1)
    c = _one(_result(_step()), index=2)  # older gym image: no counters, no stop reason
    pad = Sample(index=3, group_index=1, remove_sample=True, metadata={"mode": "failed"})
    assert _stop_metrics([[a, b], [c, pad]]) == {
        "rollout/stop/completed": pytest.approx(1 / 3),
        "rollout/stop/timeout": pytest.approx(1 / 3),
        "rollout/stop/unknown": pytest.approx(1 / 3),
        "rollout/clipped_turns": pytest.approx(2 / 3),
        "rollout/masked_output_tokens": pytest.approx(40 / 3),
    }
    assert _stop_metrics([]) == {}
    assert _stop_metrics([[pad]]) == {}


def test_removal_reason_counts_aggregates_metadata() -> None:
    def sample(remove: bool, **meta: object) -> Sample:
        return Sample(remove_sample=remove, metadata=meta)

    groups = [
        [
            sample(True, removal_reason="context_overflow"),
            sample(True, removal_reason="bad_logprobs"),
            sample(False, agent_stop_reason="timeout"),
        ],
        [sample(True, mode="failed"), sample(True), sample(False)],
    ]
    assert _removal_reason_counts(groups) == {"context_overflow": 1, "bad_logprobs": 1, "failed": 1, "other": 1}
    assert _removal_reason_counts([]) == {}


# (d) the plugin never sets advantage_scale; an old gym's spans are ignored ---
def test_plugin_never_sets_advantage_scale() -> None:
    s = _one(_result({**_step(stop_reason="length"), "truncated_spans": [[3, 6]]}, agent_stop_reason="truncated"))
    assert s.advantage_scale is None
    assert s.loss_mask == _RESPONSE
    assert s.remove_sample is False


@pytest.mark.parametrize(
    "flag",
    [
        "--arena-mask-clipped-final-turn",
        "--arena-keep-timeout-trajectories",
        "--arena-keep-context-error-trajectories",
        "--arena-truncated-turn-rule=mask",
        "--arena-truncated-turn-lambda=0.5",
        "--arena-truncated-turn-min-adv=-1",
    ],
)
def test_removed_flags_are_rejected(flag: str) -> None:
    parser = argparse.ArgumentParser()
    _add_arena_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([flag])
