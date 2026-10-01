"""training-runs/harbor-rl-glm53-flash: each run folder keeps its own identity.

A shared workflow ``experiment-name`` gives two runs the same ``--load`` and
``--save`` dir (``scripts/run_arena_harbor.py`` ``ckpt_dir``), so the second
run resumes from the first run's checkpoints and saves into them. A shared
``arena_sample_summary_dir`` makes ``_write_sample_summary`` replace the
``rollout_<id>.jsonl`` files of the other run. guparpit-agentic-debt-v2 is
the r48 config (guparpit-agentic-debt-v1) with a new identity and the
triton MoE runner. R3 needs that runner on the SGLang of the v2 trainer
image; the r48 SGLang selected a capture runner by itself.

Run: python -m pytest tests/fast/plugins/arena/test_training_run_identity.py -v
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from pathlib import Path

import yaml

FAMILY = Path(__file__).resolve().parents[4] / "training-runs" / "harbor-rl-glm53-flash"
_R47_SUMMARY_DIR = "/mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-adebt-766-r47/sample_summary"
_IDENTITY_KEYS = {"experiment_name", "project_name", "arena_sample_summary_dir"}


def _shared(pattern: str, read: Callable[[dict], str | None]) -> dict[str, set[str]]:
    """Map each value that more than one run folder uses to those folders."""
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for path in FAMILY.glob(pattern):
        if (value := read(yaml.safe_load(path.read_text()) or {})) is not None:
            owners[value].add(path.parent.name)
    return {value: runs for value, runs in owners.items() if len(runs) > 1}


def _experiment_name(workflow: dict) -> str | None:
    parameters = workflow.get("spec", {}).get("arguments", {}).get("parameters", [])
    return {p["name"]: p.get("value") for p in parameters}.get("experiment-name")


def test_each_run_folder_has_its_own_experiment_name() -> None:
    assert _shared("*/workflow*.yaml", _experiment_name) == {}


def test_each_run_folder_has_its_own_sample_summary_dir() -> None:
    # r48 kept the r47 dir (guparpit-agentic-debt-v1/miles-config.yaml). No other run may join them.
    shared = _shared("*/miles-config*.yaml", lambda cfg: cfg.get("arena_sample_summary_dir"))
    assert shared == {_R47_SUMMARY_DIR: {"r47", "guparpit-agentic-debt-v1"}}


def test_agentic_debt_v2_is_r48_with_a_new_identity() -> None:
    r48, v2 = (
        yaml.safe_load((FAMILY / run / "miles-config.yaml").read_text())
        for run in ("guparpit-agentic-debt-v1", "guparpit-agentic-debt-v2")
    )
    changed = {key for key in r48.keys() | v2.keys() if r48.get(key) != v2.get(key)}
    assert changed == _IDENTITY_KEYS | {"sglang_moe_runner_backend"}
    assert all("guparpit-agentic-debt-v2" in v2[key] for key in _IDENTITY_KEYS)
    assert "sglang_moe_runner_backend" not in r48
    assert v2["sglang_moe_runner_backend"] == "triton"
