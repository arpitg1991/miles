"""Tests that gym weight_version flows onto Sample.weight_versions.

Covers the GenerateClient fast path in ``_result_to_samples_full_trajectory``
(``has_generate_tokens`` steps carry real token-level data, so no tokenizer is
needed). The slow path needs a real tokenizer and is exercised via the metric
test + e2e. Run:
    python -m pytest tests/fast/plugins/arena/test_weight_versions.py -v
"""

from __future__ import annotations

import sys

import pytest

from miles_plugins.arena.nats_arena.nats_rollout import (
    _result_to_samples_full_trajectory,
)


def _result(steps: list[dict], weight_versions: list | None = None) -> dict:
    traj: dict = {"reward": 1.0, "steps": steps}
    if weight_versions is not None:
        traj["weight_versions"] = weight_versions
    return {
        "task_id": "t1",
        "gym_name": "g",
        "trajectories": [traj],
    }


def _gen_step(weight_version=None, n_resp=2, is_last=True):
    # Fast path keys off steps[-1].has_generate_tokens and reads its cumulative
    # token_ids/loss_mask. prompt_len=1, response_len=n_resp via loss_mask.
    step = {
        "token_ids": [0] * (1 + n_resp),
        "loss_mask": [0] + [1] * n_resp,
        "log_probs": [0.0] * n_resp,
        "stop_reason": "stop",
        "has_generate_tokens": is_last,
    }
    if weight_version is not None:
        step["weight_version"] = weight_version
    return step


def _convert(result):
    # Fast path never touches the tokenizer/args, so None is safe here.
    return _result_to_samples_full_trajectory(result, tokenizer=None, args=None)


def test_fast_path_uses_parallel_weight_versions_list():
    # The gym's explicit parallel list wins over per-step keys.
    result = _result([_gen_step()], weight_versions=[3, 4])
    samples = _convert(result)
    assert len(samples) == 1
    assert samples[0].weight_versions == ["3", "4"]


def test_fast_path_falls_back_to_per_step_versions():
    result = _result([_gen_step(weight_version=7)])
    samples = _convert(result)
    assert len(samples) == 1
    assert samples[0].weight_versions == ["7"]


def test_fast_path_missing_version_leaves_list_empty():
    result = _result([_gen_step(weight_version=None)])
    samples = _convert(result)
    assert len(samples) == 1
    assert samples[0].weight_versions == []


def test_fast_path_versions_coerced_to_str():
    result = _result([_gen_step()], weight_versions=[7])
    samples = _convert(result)
    assert samples[0].weight_versions == ["7"]


def test_fast_path_attaches_group_metrics_to_first_sample_only():
    result = _result([_gen_step()], weight_versions=[1])
    result["group_metrics"] = {"group.reward.mean": 1.0}
    samples = _convert(result)
    assert samples[0].metadata.get("group_metrics") == {"group.reward.mean": 1.0}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
