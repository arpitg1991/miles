"""--arena-keep-timeout-trajectories: keep Harbor agent-timeout trajectories
whose recorded turns all ended cleanly as training samples.

The gym worker maps the Harbor agent's per-task wall-clock AgentTimeoutError
to ``agent_stop_reason="timeout"``; every recorded turn still ended with
finish_reason "stop", the token stream ends at the model's turn-close token and
the verifier ran, so the reward is real. Default (flag off) keeps the
r5-lineage behavior: "timeout" is a degenerate agent stop -> TRUNCATED +
remove_sample=True. With the flag on the sample is kept (loss_mask untouched,
status still TRUNCATED so truncated_ratio stays honest) unless it ALSO has a
clipped final turn, a hard context overflow or zero-filled logprobs.
"""

import logging
from types import SimpleNamespace

import pytest

from miles_plugins.arena.nats_arena.nats_rollout import (
    _DEGENERATE_AGENT_STOP,
    _removal_reason_counts,
    _result_to_samples_full_trajectory,
)
from miles.utils.types import Sample

_LOGGER = "miles_plugins.arena.nats_arena.nats_rollout"


def _result(step: dict) -> dict:
    return {
        "task_id": "t1",
        "gym_name": "g",
        "trajectories": [{"reward": 1.0, "steps": [step], **step.pop("_traj", {})}],
    }


def _step(
    loss_mask: list[int],
    stop_reason: str = "stop",
    traj_extra: dict | None = None,
    log_probs: list[float] | None = None,
) -> dict:
    # Cumulative fast-path arrays; log_probs aligned 1:1 with token_ids so the
    # rollout-logprob check stays green unless a test overrides them.
    step = {
        "token_ids": [7] * len(loss_mask),
        "loss_mask": list(loss_mask),
        "log_probs": [-0.1] * len(loss_mask) if log_probs is None else log_probs,
        "stop_reason": stop_reason,
        "has_generate_tokens": True,
    }
    if traj_extra:
        step["_traj"] = traj_extra
    return step


def _args(keep_timeout: bool, mask_clipped: bool = False, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        arena_keep_timeout_trajectories=keep_timeout,
        arena_mask_clipped_final_turn=mask_clipped,
        **extra,
    )


# Multi-turn trajectory: prompt, turn A (2 toks), tool output, turn B (3 toks).
# Response window starts at the first 1. Every turn ended cleanly (the token
# stream ends at the model's turn-close token) -- the agent simply ran out of
# wall-clock before the loop finished.
_MULTI_TURN = [0, 1, 1, 0, 1, 1, 1]
_RESPONSE = _MULTI_TURN[1:]  # [1, 1, 0, 1, 1, 1]
_TIMEOUT = {"agent_stop_reason": "timeout"}


def _timeout_step(stop_reason: str = "stop", **kw) -> dict:
    return _step(_MULTI_TURN, stop_reason=stop_reason, traj_extra=dict(_TIMEOUT), **kw)


# (a) flag off: current behavior preserved -----------------------------------
def test_flag_off_timeout_still_removed():
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step()), tokenizer=None, args=_args(False)
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.loss_mask == _RESPONSE  # untouched; remove_sample does the zeroing
    assert s.metadata["agent_stop_reason"] == "timeout"
    assert s.metadata["removal_reason"] == "timeout"
    assert "kept_timeout" not in s.metadata


def test_legacy_caller_without_args_still_removes_timeout():
    # test_weight_versions-style callers pass args=None; getattr default keeps
    # them on the r5-lineage behavior.
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step()), tokenizer=None, args=None
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True


# (b) flag on, clean multi-turn timeout: kept --------------------------------
def test_flag_on_keeps_clean_timeout_trajectory(caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step()), tokenizer=None, args=_args(True)
    )
    assert s.status == Sample.Status.TRUNCATED  # truncated_ratio stays honest
    assert getattr(s, "remove_sample", False) is False
    assert s.loss_mask == _RESPONSE  # no zeroing at all
    assert s.response_length == len(s.loss_mask)
    assert len(s.rollout_log_probs) == s.response_length
    assert s.reward == 1.0
    assert s.metadata["kept_timeout"] is True
    assert s.metadata["agent_stop_reason"] == "timeout"
    assert "removal_reason" not in s.metadata
    # One info line per salvaged sample, with turn (= 1-run) and token counts.
    kept_lines = [r for r in caplog.records if "kept timeout trajectory" in r.getMessage()]
    assert len(kept_lines) == 1
    assert kept_lines[0].levelno == logging.INFO
    assert "(2 turns, 6 response tokens)" in kept_lines[0].getMessage()


# (c) flag on, other degenerate stops: still removed -------------------------
@pytest.mark.parametrize("reason", sorted(_DEGENERATE_AGENT_STOP - {"timeout"}))
def test_flag_on_other_degenerate_stops_still_removed(reason):
    (s,) = _result_to_samples_full_trajectory(
        _result(_step(_MULTI_TURN, traj_extra={"agent_stop_reason": reason})),
        tokenizer=None,
        args=_args(True),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.loss_mask == _RESPONSE
    assert s.metadata["removal_reason"] == reason
    assert "kept_timeout" not in s.metadata


# (d) flag on, timeout + clipped final turn, mask-clipped OFF: removed -------
def test_flag_on_timeout_with_clipped_turn_removed_when_mask_clipped_off():
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step(stop_reason="length")),
        tokenizer=None,
        args=_args(True, mask_clipped=False),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.loss_mask == _RESPONSE  # untouched; removal zeroes it downstream
    # The agent stop is reported as the reason (the dominant, unsalvageable
    # defect from this flag's point of view), not "length".
    assert s.metadata["removal_reason"] == "timeout"
    assert "kept_timeout" not in s.metadata


# (e) flag on + mask-clipped on, timeout + clipped final turn: salvaged ------
def test_both_flags_timeout_with_clipped_turn_masks_only_final_turn(caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step(stop_reason="length")),
        tokenizer=None,
        args=_args(True, mask_clipped=True),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert getattr(s, "remove_sample", False) is False
    # Turn A survives, turn B (trailing 1-run) zeroed -- the timeout no longer
    # blocks the mask-clipped salvage.
    assert s.loss_mask == [1, 1, 0, 0, 0, 0]
    assert s.response_length == len(s.loss_mask)
    assert len(s.rollout_log_probs) == s.response_length
    assert "removal_reason" not in s.metadata
    # Salvage came from the mask-clipped path, not the keep-timeout branch.
    assert not s.metadata.get("kept_timeout")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("masked clipped final turn" in m for m in msgs)
    assert not any("kept timeout trajectory" in m for m in msgs)


def test_mask_clipped_alone_still_refuses_timeout():
    # Without --arena-keep-timeout-trajectories the degenerate stop still
    # blocks the mask-clipped salvage (unchanged r5-lineage precedence).
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step(stop_reason="length")),
        tokenizer=None,
        args=_args(False, mask_clipped=True),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.loss_mask == _RESPONSE


# (f) clean stop unaffected ---------------------------------------------------
@pytest.mark.parametrize("agent_stop", [None, "completed", "max_iterations"])
def test_flag_on_clean_stop_untouched(agent_stop):
    extra = {"agent_stop_reason": agent_stop} if agent_stop else None
    (s,) = _result_to_samples_full_trajectory(
        _result(_step(_MULTI_TURN, traj_extra=extra)), tokenizer=None, args=_args(True)
    )
    assert s.status == Sample.Status.COMPLETED
    assert getattr(s, "remove_sample", False) is False
    assert s.loss_mask == _RESPONSE
    assert "removal_reason" not in s.metadata
    assert "kept_timeout" not in s.metadata
    assert s.metadata.get("agent_stop_reason") == agent_stop


# Other defects on a timeout trajectory are never salvaged -------------------
def test_flag_on_timeout_with_empty_response_still_removed():
    # Nothing trainable: has_response is False -> keep branch must not fire.
    (s,) = _result_to_samples_full_trajectory(
        _result(_step([0, 0, 0, 0], traj_extra=dict(_TIMEOUT))),
        tokenizer=None,
        args=_args(True),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.loss_mask == []
    assert "kept_timeout" not in s.metadata


def test_flag_on_timeout_with_hard_context_overflow_still_removed():
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step()),
        tokenizer=None,
        args=_args(True, rollout_max_context_len=4),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert len(s.tokens) == 4
    assert s.metadata["removal_reason"] == "context_overflow"
    assert "kept_timeout" not in s.metadata


def test_flag_on_timeout_with_bad_logprobs_still_removed():
    (s,) = _result_to_samples_full_trajectory(
        _result(_timeout_step(log_probs=[-0.1] * 3)),  # misaligned with tokens
        tokenizer=None,
        args=_args(True),
    )
    assert s.status == Sample.Status.ABORTED
    assert s.remove_sample is True
    assert s.rollout_log_probs == [0.0] * s.response_length
    assert s.metadata["removal_reason"] == "bad_logprobs"
    assert "kept_timeout" not in s.metadata


# Per-rollout "Removal reasons" summary --------------------------------------
def _sample(remove: bool, **meta) -> Sample:
    s = Sample()
    s.remove_sample = remove
    s.metadata = meta
    return s


def test_removal_reason_counts_aggregates_metadata():
    groups = [
        [
            _sample(True, removal_reason="timeout"),
            _sample(True, removal_reason="timeout"),
            _sample(True, removal_reason="context_error"),
            _sample(False, kept_timeout=True),
        ],
        [
            _sample(True, removal_reason="length"),
            _sample(True, mode="failed"),  # gym-side pad
            _sample(True),  # unattributed removal
            _sample(False),
        ],
    ]
    reasons, kept = _removal_reason_counts(groups)
    assert reasons == {
        "timeout": 2,
        "context_error": 1,
        "length": 1,
        "failed": 1,
        "other": 1,
    }
    assert kept == 1


def test_removal_reason_counts_empty():
    assert _removal_reason_counts([]) == ({}, 0)
