"""training-runs/build_index.py: the Runs table of INDEX.md from the RECORD.md headers.

Run: python -m pytest --confcutdir=tests/fast/plugins/arena tests/fast/plugins/arena/test_training_runs_index.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNS = REPO_ROOT / "training-runs"

_spec = importlib.util.spec_from_file_location("build_index", RUNS / "build_index.py")
build_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_index)

_RECORD = """# Run record: {run} — test

**Status:** {status}
<!-- gen-workflow:begin -->
**Date:** 2026-09-27
**Family:** `fam`
**Dataset:** `lakefs://arena-inspect/dev/x/manifest.jsonl` (gym `x`)
**Gym image:** `123.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-1`
**Trainer image:** `123.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-1`
<!-- gen-workflow:end -->
**Outcome:** First 0.5 at 27.5 h. Second | pipe. Third is dropped.

## Goal
"""


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "training-runs"
    for run, status in (("r1", "Retired"), ("r2", "Running"), ("r2b", "Prepared")):
        (root / "fam" / run).mkdir(parents=True)
        (root / "fam" / run / "RECORD.md").write_text(_RECORD.format(run=run, status=status))
    (root / "fam" / "r3").mkdir()  # a legacy folder without a record
    (root / "INDEX.md").write_text(f"# Index\n\nintro\n\n{build_index.BEGIN}\nold\n{build_index.END}\n\n## Studies\n\nkept\n")
    return root


def test_rows_newest_first_and_text_outside_markers_kept(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert build_index.main(["--root", str(root)]) == 0
    text = (root / "INDEX.md").read_text()
    assert text.startswith("# Index\n\nintro\n\n")
    assert text.endswith("\n\n## Studies\n\nkept\n")
    assert "old" not in text
    lines = [line for line in text.splitlines() if line.startswith("| r")]
    assert [line.split(" | ")[0] for line in lines] == ["| r2b", "| r2", "| r1"]
    assert lines[1] == (
        "| r2 | `fam` | 2026-09-27 | `lakefs://arena-inspect/dev/x/manifest.jsonl` (gym `x`) "
        "| `arena-slime-dev:gym-1`; `arena-slime-dev:miles-1` | Running | First 0.5 at 27.5 h. Second \\| pipe. "
        "| [r2/RECORD.md](fam/r2/RECORD.md) |"
    )


def test_named_run_folders_sort_by_date_then_number(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    for run in ("guparpit-x-v9", "guparpit-x-v10"):
        (root / "fam" / run).mkdir()
        record = _RECORD.format(run=run, status="Prepared").replace("2026-09-27", "2026-09-30")
        (root / "fam" / run / "RECORD.md").write_text(record)
    build_index.main(["--root", str(root)])
    lines = [line for line in (root / "INDEX.md").read_text().splitlines() if "| `fam` |" in line]
    assert [line.split(" | ")[0] for line in lines] == ["| guparpit-x-v10", "| guparpit-x-v9", "| r2b", "| r2", "| r1"]


def test_check_reports_a_stale_table(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert build_index.main(["--root", str(root), "--check"]) == 1
    build_index.main(["--root", str(root)])
    assert build_index.main(["--root", str(root), "--check"]) == 0


def test_committed_index_is_current() -> None:
    assert build_index.main(["--root", str(RUNS), "--check"]) == 0, "run: python training-runs/build_index.py"
