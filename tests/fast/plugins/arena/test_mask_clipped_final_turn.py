"""--arena-mask-clipped-final-turn: salvage trajectories whose FINAL generate
hit the per-turn token cap by zeroing only that turn's loss-mask tokens.

The gym ships one step per trajectory whose ``stop_reason`` is "length"
exactly when the final generate truncated; the final turn's output tokens are
the trailing run of 1s in the cumulative loss_mask. Default (flag off) keeps
the r5-lineage behavior: the whole sample is removed from the loss.
"""

from types import SimpleNamespace

from miles_plugins.arena.nats_arena.nats_rollout import (
    _result_to_samples_full_trajectory,
)
from miles.utils.types import Sample


def _result(step: dict) -> dict:
    return {
        "task_id": "t1",
        "gym_name": "g",
        "trajectories": [{"reward": 1.0, "steps": [step], **step.pop("_traj", {})}],
    }


def _step(loss_mask: list[int], stop_reason: str = "length", traj_extra: dict | None = None) -> dict:
    # Cumulative fast-path arrays; log_probs aligned 1:1 with token_ids so the
    # rollout-logprob check stays green.
    step = {
        "token_ids": [7] * len(loss_mask),
        "loss_mask": list(loss_mask),
        "log_probs": [-0.1] * len(loss_mask),
        "stop_reason": stop_reason,
        "has_generate_tokens": True,
    }
    if traj_extra:
        step["_traj"] = traj_extra
    return step


def _args(flag: bool) -> SimpleNamespace:
    # Carry both arena salvage knobs explicitly (the read sites use
    # getattr-with-default, so older callers without them still work).
    return SimpleNamespace(
        arena_mask_clipped_final_turn=flag,
        arena_keep_timeout_trajectories=False,
    )


# Multi-turn trajectory: prompt, turn A (2 toks), tool output, turn B (3 toks,
# clipped). Response window starts at the first 1.
_MULTI_TURN = [0, 1, 1, 0, 1, 1, 1]


def test_flag_off_keeps_removal():
    (s,) = _result_to_samples_full_trajectory(
        _result(_step(_MULTI_TURN)), tokenizer=None, args=_args(False)
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True


def test_flag_on_masks_only_final_turn():
    (s,) = _result_to_samples_full_trajectory(
        _result(_step(_MULTI_TURN)), tokenizer=None, args=_args(True)
    )
    assert s.status == Sample.Status.TRUNCATED  # truncated_ratio stays honest
    assert getattr(s, "remove_sample", False) is False
    # Response window = mask[1:]; turn A survives, turn B (trailing 1-run) zeroed.
    assert s.loss_mask == [1, 1, 0, 0, 0, 0]
    assert s.response_length == len(s.loss_mask)
    assert len(s.rollout_log_probs) == s.response_length


def test_flag_on_single_turn_fully_clipped_still_removed():
    # The whole response is the clipped turn: nothing left to train on.
    (s,) = _result_to_samples_full_trajectory(
        _result(_step([0, 1, 1, 1])), tokenizer=None, args=_args(True)
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True


def test_flag_on_degenerate_agent_stop_still_removed():
    # context_error etc. are mangled conversations, not per-turn clips.
    (s,) = _result_to_samples_full_trajectory(
        _result(
            _step(
                _MULTI_TURN,
                stop_reason="stop",
                traj_extra={"agent_stop_reason": "context_error"},
            )
        ),
        tokenizer=None,
        args=_args(True),
    )
    assert s.status == Sample.Status.TRUNCATED
    assert s.remove_sample is True
    assert s.loss_mask == [1, 1, 0, 1, 1, 1]  # untouched


def test_flag_on_clean_stop_untouched():
    (s,) = _result_to_samples_full_trajectory(
        _result(_step(_MULTI_TURN, stop_reason="stop")), tokenizer=None, args=_args(True)
    )
    assert s.status == Sample.Status.COMPLETED
    assert getattr(s, "remove_sample", False) is False
    assert s.loss_mask == [1, 1, 0, 1, 1, 1]
