"""ArenaDataSourceWithBuffer: skip prompts whose recorded reward exceeds a threshold.

Run: python -m pytest tests/fast/plugins/arena/test_skip_prompt_above_reward.py -v
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

pytest.importorskip("torch")

from miles.utils.types import Sample  # noqa: E402
from miles_plugins.arena.nats_arena.data_source import (  # noqa: E402
    ArenaDataSourceWithBuffer,
    _read_manifest,
)
from miles_plugins.arena.nats_arena.nats_rollout import _prompt_mean_rewards  # noqa: E402

GYM = "g"
N = 8


class _FakeDataset:
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def shuffle(self, seed: int) -> None:
        pass


def _make_ds(tmp_path, threshold, save_dir=None) -> ArenaDataSourceWithBuffer:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"lakefs_uri": f"lakefs://r/b/tasks/t{i}/", "lakefs_commit_id": "c"}) + "\n"
            for i in range(N)
        )
    )
    samples = []
    for row in _read_manifest(str(manifest)):
        s = Sample()
        s.metadata = dict(row["metadata"])
        samples.append(s)

    # Bypass __init__ (tokenizer + lakeFS); set only what the read/save path uses.
    ds = object.__new__(ArenaDataSourceWithBuffer)
    ds.args = SimpleNamespace(
        n_samples_per_prompt=2, rollout_shuffle=False, save=save_dir, load=save_dir,
    )
    ds.sample_group_index = 0
    ds.sample_index = 0
    ds.metadata = {}
    ds.prompt_reward = {}
    ds.skip_prompt_above_reward = threshold
    ds._skipped_this_epoch = {}
    ds.gym_names = [GYM]
    ds.datasets = {GYM: _FakeDataset(samples)}
    ds.offsets = {GYM: 0}
    ds.epochs = {GYM: 0}
    ds.weights = {GYM: 1.0}
    ds._weights_lock = threading.Lock()
    return ds


def _ids(groups: list[list[Sample]]) -> list[str]:
    return [g[0].metadata["instance_id"] for g in groups]


def _group(iid: str, rewards: list[float]) -> list[Sample]:
    out = []
    for i, r in enumerate(rewards):
        s = Sample()
        s.group_index = 0
        s.index = i
        s.reward = r
        s.metadata = {"instance_id": iid, "mode": "full_trajectory"}
        out.append(s)
    return out


HIGH = {f"t{i}": 1.0 for i in range(5)}
LOW = {f"t{i}": 0.25 for i in range(5, 8)}


def test_epoch1_skips_high_reward_prompts(tmp_path):
    ds = _make_ds(tmp_path, 0.5)
    assert _ids(ds._read_from_gym(GYM, N)) == [f"t{i}" for i in range(N)]
    assert ds.epochs[GYM] == 1

    ds.record_prompt_rewards({**HIGH, **LOW})
    assert _ids(ds._read_from_gym(GYM, 3)) == ["t5", "t6", "t7"]
    # The pass wrapped: epoch 2 starts, the 5 high prompts were skipped.
    assert ds.epochs[GYM] == 2
    assert ds.offsets[GYM] == 0


def test_threshold_none_emits_all(tmp_path):
    ds = _make_ds(tmp_path, None)
    ds._read_from_gym(GYM, N)
    ds.record_prompt_rewards({**HIGH, **LOW})
    assert len(_ids(ds._read_from_gym(GYM, N))) == N


def test_all_skipped_falls_back(tmp_path):
    ds = _make_ds(tmp_path, 0.5)
    ds._read_from_gym(GYM, N)
    ds.record_prompt_rewards({f"t{i}": 1.0 for i in range(N)})
    assert len(_ids(ds._read_from_gym(GYM, N))) == N


def test_save_load_roundtrip(tmp_path):
    ds = _make_ds(tmp_path, 0.5, save_dir=str(tmp_path / "ckpt"))
    ds._read_from_gym(GYM, N)
    ds.record_prompt_rewards({**HIGH, **LOW})
    ds.save(7)

    ds2 = _make_ds(tmp_path, 0.5, save_dir=str(tmp_path / "ckpt"))
    ds2.load(7)
    assert ds2.prompt_reward == {**HIGH, **LOW}
    assert ds2.epochs[GYM] == 1
    assert _ids(ds2._read_from_gym(GYM, 3)) == ["t5", "t6", "t7"]


def test_prompt_mean_rewards_uses_episodes():
    m = _prompt_mean_rewards([_group("a", [1.0, 1.0]), _group("b", [1.0, 0.0, 0.0, 0.0])])
    assert m == {"a": 1.0, "b": 0.25}
