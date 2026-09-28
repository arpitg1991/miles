"""examples/arena/harbor-rl-glm53-flash/gen-workflow.py: workflow.yaml and the RECORD.md header.

The run folders live in training-runs/<family>/r<N>/ (training-runs/README.md).
The script copies the base parameters, drops the retired ones, keeps the
miles-config block byte for byte, and writes the RECORD.md header between the
gen-workflow markers of training-runs/TEMPLATE.md. A re-run keeps the text
outside the markers and the first Date. A config without prompt-data-list fails
before any file is written.

Run: python -m pytest tests/fast/plugins/arena/test_gen_workflow.py -v
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
FAMILY = "harbor-rl-glm53-flash"
SCRIPT = REPO_ROOT / "examples" / "arena" / FAMILY / "gen-workflow.py"
TEMPLATE = REPO_ROOT / "training-runs" / "TEMPLATE.md"

_spec = importlib.util.spec_from_file_location("gen_workflow", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

_BASE_WORKFLOW = {
    "apiVersion": "argoproj.io/v1alpha1",
    "kind": "Workflow",
    "metadata": {"generateName": "rl-glm53f1-", "namespace": "arena-tasks"},
    "spec": {
        "workflowTemplateRef": {"name": "guparpit-miles-deployer-v9"},
        "arguments": {
            "parameters": [
                {"name": "experiment-name", "value": "rl-glm53f-gbash-r1"},
                {"name": "trainer-image", "value": "ecr/arena-slime-dev:miles-glm53-r1"},
                {"name": "compaction-max", "value": "4"},
                {"name": "partial-reward", "value": "ctrf"},
                {"name": "miles-config", "value": "x: 1\n"},
                {"name": "gym-image", "value": "ecr/arena-slime-dev:gym-r1"},
            ]
        },
    },
}
_DATASET = "lakefs://arena-inspect/327e057c/internal/agentic-debt-766/manifest.jsonl"
_CONFIG = (
    "# r2: the test config\n"
    "experiment_name: rl-glm53f-test-r2\n"
    "wandb_project: rl-glm53f-test\n"
    "lr: 1.5e-06\n"
    f"prompt-data-list: '[{{\"path\":\"{_DATASET}\",\"gym_name\":\"agentic-debt\",\"weight\":1.0}}]'\n"
)


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "training-runs"
    fam = root / FAMILY
    (fam / "r1").mkdir(parents=True)
    (fam / "r2").mkdir()
    shutil.copy(TEMPLATE, root / "TEMPLATE.md")
    (fam / "r1" / "workflow.yaml").write_text(yaml.dump(_BASE_WORKFLOW, sort_keys=False))
    (fam / "r2" / "miles-config.yaml").write_text(_CONFIG)
    return fam


def _run(runs: Path, *extra: str) -> None:
    gen.main(
        [
            "2",
            "--base",
            "r1",
            "--runs-dir",
            str(runs),
            "--template",
            "guparpit-miles-deployer-v10",
            "--gym-image",
            "ecr/arena-slime-dev:gym-r2",
            *extra,
        ]
    )


def _params(runs: Path) -> dict[str, str]:
    wf = yaml.safe_load((runs / "r2" / "workflow.yaml").read_text())
    return {p["name"]: p["value"] for p in wf["spec"]["arguments"]["parameters"]}


def test_default_runs_dir_is_training_runs() -> None:
    assert gen.RUNS_DIR == REPO_ROOT / "training-runs" / FAMILY
    assert (gen.RUNS_DIR / "r47" / "workflow.yaml").exists()


def test_workflow_parameters(runs: Path) -> None:
    _run(runs)
    got = _params(runs)
    assert got["miles-config"] == _CONFIG
    assert got["experiment-name"] == "rl-glm53f-gbash-r2"
    assert got["gym-image"] == "ecr/arena-slime-dev:gym-r2"
    assert got["trainer-image"] == "ecr/arena-slime-dev:miles-glm53-r1"
    assert got["agent-kwargs"] == '{"max_compactions": 4}'
    assert "partial-reward" not in got
    assert "compaction-max" not in got
    wf = yaml.safe_load((runs / "r2" / "workflow.yaml").read_text())
    assert wf["metadata"]["generateName"] == "rl-glm53f2-"
    assert wf["spec"]["workflowTemplateRef"]["name"] == "guparpit-miles-deployer-v10"


def test_record_header_from_template(runs: Path) -> None:
    _run(runs)
    text = (runs / "r2" / "RECORD.md").read_text()
    assert text.startswith("# Run record: r2 ")
    assert re.search(r"^\*\*Date:\*\* \d{4}-\d{2}-\d{2}$", text, re.M)
    for line in (
        "**Family:** `harbor-rl-glm53-flash`",
        "**Argo generateName:** `rl-glm53f2-`",
        "**Experiment name:** `rl-glm53f-gbash-r2`",
        "**W&B project:** `rl-glm53f-test` (group `rl-glm53f-gbash-r2`)",
        f"**Dataset:** `{_DATASET}` (gym `agentic-debt`)",
        "**Manifest commit:** `327e057c`",
        "**Gym image:** `ecr/arena-slime-dev:gym-r2`",
        "**Trainer image:** `ecr/arena-slime-dev:miles-glm53-r1`",
        "**Template:** `guparpit-miles-deployer-v10`",
        "**Base:** `r1`",
        "**Argo workflow:** `rl-glm53f2-<suffix>`",
        "| Item | Base r1 | This run |",
    ):
        assert line in text, line
    assert "r<N>" not in text
    assert "r<M>" not in text
    for section in ("## Goal", "## Setup", "## Timeline", "## Results", "## Issues", "## Follow-ups", "## Sources"):
        assert section in text


def test_rerun_keeps_text_outside_markers_and_the_date(runs: Path) -> None:
    _run(runs)
    record = runs / "r2" / "RECORD.md"
    text = re.sub(r"\*\*Date:\*\* \S+", "**Date:** 2026-01-02", record.read_text(), count=1)
    text = text.replace("**Status:** Prepared | Running | Complete | Retired | Invalid", "**Status:** Running")
    record.write_text(text + "\nHand-written note.\n")
    _run(runs, "--trainer-image", "ecr/arena-slime-dev:miles-glm53-r2")
    new = record.read_text()
    assert "**Date:** 2026-01-02" in new
    assert "**Status:** Running" in new
    assert new.endswith("\nHand-written note.\n")
    assert "**Trainer image:** `ecr/arena-slime-dev:miles-glm53-r2`" in new
    assert "miles-glm53-r1`" not in new
    assert new.count(gen.RECORD_BEGIN) == 1
    assert new.count(gen.RECORD_END) == 1


def test_local_manifest_has_no_lakefs_ref(runs: Path) -> None:
    local = "/mnt/scratch-s3files-rw/guparpit/data/agentic-debt/manifest-le5.jsonl"
    (runs / "r2" / "miles-config.yaml").write_text(_CONFIG.replace(_DATASET, local))
    _run(runs)
    text = (runs / "r2" / "RECORD.md").read_text()
    assert f"**Dataset:** `{local}` (gym `agentic-debt`)" in text
    assert "**Manifest commit:** none (not a lakeFS URI)" in text


def test_missing_dataset_fails_before_any_write(runs: Path) -> None:
    (runs / "r2" / "miles-config.yaml").write_text("experiment_name: rl-glm53f-test-r2\n")
    with pytest.raises(SystemExit, match="FAIL"):
        _run(runs)
    assert not (runs / "r2" / "workflow.yaml").exists()
    assert not (runs / "r2" / "RECORD.md").exists()


def test_record_without_markers_fails(runs: Path) -> None:
    (runs / "r2" / "RECORD.md").write_text("# hand-written record without markers\n")
    with pytest.raises(SystemExit, match="FAIL"):
        _run(runs)
