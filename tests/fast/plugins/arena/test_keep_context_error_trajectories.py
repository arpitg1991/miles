"""--arena-keep-context-error-trajectories: keep trajectories the arena
Terminus-2 agent ended on a context-window overflow as training samples.

The gym stamps ``agent_stop_reason="context_error"`` when the engine rejected
the NEXT turn's prompt; every recorded turn ended cleanly and the verifier ran.
Default (flag off) keeps the old behavior: TRUNCATED + remove_sample=True.
With the flag on the sample is kept (loss_mask untouched, status still
TRUNCATED) unless it also has a clipped final turn, a hard overflow or
zero-filled logprobs. The timeout flag alone does NOT admit context errors.
"""

from types import SimpleNamespace

from miles_plugins.arena.nats_arena.nats_rollout import (
    _removal_reason_counts,
    _result_to_samples_full_trajectory,
)
from miles.utils.types import Sample

_MULTI_TURN = [0, 1, 1, 0, 1, 1, 1]


def _result(stop_reason: str = "stop", log_probs=None, agent_stop="context_error"):
    step = {
        "token_ids": [7] * len(_MULTI_TURN),
        "loss_mask": list(_MULTI_TURN),
        "log_probs": [-0.1] * len(_MULTI_TURN) if log_probs is None else log_probs,
        "stop_reason": stop_reason,
        "has_generate_tokens": True,
    }
    return {
        "task_id": "t1",
        "gym_name": "g",
        "trajectories": [{"reward": 1.0, "steps": [step], "agent_stop_reason": agent_stop}],
    }


def _args(keep_ctx: bool, keep_timeout: bool = False, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        arena_keep_context_error_trajectories=keep_ctx,
        arena_keep_timeout_trajectories=keep_timeout,
        arena_mask_clipped_final_turn=False,
        **extra,
    )


def test_flag_off_context_error_removed():
    (s,) = _result_to_samples_full_trajectory(_result(), tokenizer=None, args=_args(False))
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.metadata["removal_reason"] == "context_error"


def test_timeout_flag_alone_does_not_admit_context_error():
    (s,) = _result_to_samples_full_trajectory(
        _result(), tokenizer=None, args=_args(False, keep_timeout=True)
    )
    assert s.remove_sample is True


def test_flag_on_keeps_sample_truncated_with_real_reward():
    (s,) = _result_to_samples_full_trajectory(_result(), tokenizer=None, args=_args(True))
    assert s.status == Sample.Status.TRUNCATED
    assert not getattr(s, "remove_sample", False)
    assert s.reward == 1.0
    assert s.loss_mask == _MULTI_TURN[1:]
    assert s.metadata["kept_context_error"] is True
    assert "kept_timeout" not in s.metadata
    assert "removal_reason" not in s.metadata


def test_flag_on_still_removes_clipped_final_turn():
    (s,) = _result_to_samples_full_trajectory(
        _result(stop_reason="length"), tokenizer=None, args=_args(True)
    )
    assert s.remove_sample is True
    # Same precedence as the timeout suite: the agent stop is the reported
    # (dominant, unsalvageable) reason when a clipped turn blocks the salvage.
    assert s.metadata["removal_reason"] == "context_error"
    assert "kept_context_error" not in s.metadata


def test_flag_on_still_removes_bad_logprobs():
    (s,) = _result_to_samples_full_trajectory(
        _result(log_probs=[-0.1] * 3),  # misaligned with tokens
        tokenizer=None, args=_args(True),
    )
    assert s.status == Sample.Status.ABORTED
    assert s.remove_sample is True
    assert s.metadata["removal_reason"] == "bad_logprobs"
    assert "kept_context_error" not in s.metadata


def test_flag_does_not_touch_other_degenerate_stops():
    (s,) = _result_to_samples_full_trajectory(
        _result(agent_stop="context_churn"), tokenizer=None, args=_args(True)
    )
    assert s.remove_sample is True
    assert s.metadata["removal_reason"] == "context_churn"


def test_removal_reason_counts_report_kept_context_error():
    (kept,) = _result_to_samples_full_trajectory(_result(), tokenizer=None, args=_args(True))
    (removed,) = _result_to_samples_full_trajectory(_result(), tokenizer=None, args=_args(False))
    reasons, kept_timeout, kept_ctx = _removal_reason_counts([[kept, removed]])
    assert reasons == {"context_error": 1}
    assert (kept_timeout, kept_ctx) == (0, 1)
