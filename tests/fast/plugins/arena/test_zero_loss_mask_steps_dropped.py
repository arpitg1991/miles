"""All-zero loss_mask guard in ``_result_to_episodes_full_trajectory``.

A step whose loss_mask has no 1 yields ``response_length == 0``; packed into
train_data that row makes ``prompt_length == total_length`` and breaks the
Megatron loss / CP slicing. The guard drops the step, skips a trajectory with
nothing left (``_process_group`` pads the slot from a sibling, like the
synthetic path) and counts the drops as ``arena/zero_mask_steps_dropped``.

Run: python -m pytest tests/fast/plugins/arena/test_zero_loss_mask_steps_dropped.py -v
"""

from __future__ import annotations

from tests.fast.plugins.arena.test_multi_segment_episodes import (
    _drain,
    _make_args,
    _make_worker,
    _result,
    _segment,
    _step,
    _traj,
)

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import (
    _batch_telemetry,
    _result_to_episodes_full_trajectory,
)

_ZERO = [0, 0, 0, 0]
_NORMAL = [0, 0, 1, 1]


def test_zero_mask_segment_dropped_from_multi_segment_episode() -> None:
    args = _make_args(n_samples_per_prompt=1, arena_train_segments="all")
    steps = [_segment(_ZERO, tag=1), _step(_NORMAL, tag=9)]
    (episode,) = _result_to_episodes_full_trajectory(_result([_traj(steps)]), None, args)
    (s,) = episode
    assert s.response_length == 2
    assert s.metadata["segment"] == 0 and s.metadata["n_segments"] == 1
    assert s.metadata["zero_mask_steps_dropped"] == 1


def test_all_zero_trajectory_skipped_and_padded_by_process_group() -> None:
    args = _make_args(n_samples_per_prompt=2, arena_train_segments="final")
    result = _result([_traj([_step(_ZERO, tag=1)]), _traj([_step(_NORMAL, tag=9)])])

    episodes = _result_to_episodes_full_trajectory(result, None, args)
    assert len(episodes) == 1
    assert episodes[0][0].metadata["zero_mask_steps_dropped"] == 1

    worker = _make_worker(args)
    worker._process_group("t.g0.", [result])
    (group,) = _drain(worker)
    real, pad = group
    assert real.response_length == 2
    assert pad.status == Sample.Status.FAILED and pad.remove_sample is True
    assert pad.response_length == real.response_length  # a sibling copy, never a zero-length row
    assert all(s.response_length > 0 for s in group)

    tm = _batch_telemetry([group], [group])
    assert tm["zero_mask_steps_dropped"] == 1
    assert tm["failed_count"] == 1


def test_every_trajectory_zero_yields_no_episode() -> None:
    args = _make_args(n_samples_per_prompt=1, arena_train_segments="final")
    assert _result_to_episodes_full_trajectory(_result([_traj([_step(_ZERO)])]), None, args) == []
