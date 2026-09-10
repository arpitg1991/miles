"""--arena-shuffle-after-first-epoch: epoch 0 walks the manifest in file order,
every later epoch rollover reshuffles with seed = epoch, and a checkpoint
resume at epoch 1 reproduces the epoch-1 order.
"""

import json
from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

N = 32


def _make_args(tmp_path, manifest, rollout_shuffle=False):
    return SimpleNamespace(
        prompt_data_list=[{"path": str(manifest), "gym_name": "g", "weight": 1.0}],
        hf_checkpoint=None,
        rollout_max_prompt_len=None,
        input_key="messages",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=42,
        rollout_shuffle=rollout_shuffle,
        arena_shuffle_after_first_epoch=True,
        n_samples_per_prompt=1,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=str(tmp_path),
    )


def _ids(groups):
    return [g[0].metadata["instance_id"] for g in groups]


@pytest.fixture
def ds_module(monkeypatch):
    pytest.importorskip("torch")
    from miles_plugins.arena.nats_arena import data_source

    monkeypatch.setattr(data_source, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(data_source, "load_processor", lambda *a, **k: None)
    return data_source


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    with open(path, "w") as f:
        for i in range(N):
            f.write(json.dumps({"lakefs_uri": f"lakefs://r/b/tasks/t{i:03d}/", "lakefs_commit_id": "c"}) + "\n")
    return path


def test_file_order_then_reshuffle_and_resume(ds_module, manifest, tmp_path):
    expected = [f"t{i:03d}" for i in range(N)]
    ds = ds_module.ArenaDataSourceWithBuffer(_make_args(tmp_path, manifest))

    epoch0 = _ids(ds._read_from_gym("g", N))
    assert epoch0 == expected
    # The rollover at the end of epoch 0 already reshuffled for epoch 1.
    assert (ds.epochs["g"], ds.offsets["g"]) == (1, 0)
    ds.save(rollout_id=7)

    epoch1 = _ids(ds._read_from_gym("g", N))
    assert sorted(epoch1) == expected and epoch1 != expected

    # Resume from the epoch-1 checkpoint in a fresh instance reproduces the epoch-1 order.
    resumed = ds_module.ArenaDataSourceWithBuffer(_make_args(tmp_path, manifest))
    resumed.load(rollout_id=7)
    assert resumed.epochs["g"] == 1
    assert _ids(resumed._read_from_gym("g", N)) == epoch1

def test_both_flags_rejected(ds_module, manifest, tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        ds_module.ArenaDataSourceWithBuffer(_make_args(tmp_path, manifest, rollout_shuffle=True))
