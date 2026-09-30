"""Rewrite the Runs table of training-runs/INDEX.md from the <run>/RECORD.md headers.

Usage: python training-runs/build_index.py [--root training-runs] [--check]

Reads every <family>/<run>/RECORD.md under the root and takes the header
fields Status, Date, Family, Dataset, Gym image, Trainer image, and Outcome.
Then it rewrites the lines between `<!-- runs:begin -->` and
`<!-- runs:end -->` in INDEX.md; text outside the markers stays. The
Studies table of INDEX.md is hand-written. `--check` writes nothing and
exits 1 when the table is stale.
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BEGIN = "<!-- runs:begin -->"
END = "<!-- runs:end -->"
COLUMNS = ("Run", "Family", "Date", "Dataset", "Images (gym; trainer)", "Status", "Outcome", "Record")
_FIELD = re.compile(r"^\*\*(.+?):\*\* (.*)$", re.M)


def header(record: pathlib.Path) -> dict[str, str]:
    """Return the `**Key:** value` fields above the first `## ` heading."""
    return dict(_FIELD.findall(record.read_text().split("\n## ", 1)[0]))


def _image(value: str) -> str:
    # `registry/repo:tag` -> `repo:tag`; the record keeps the full name.
    return re.sub(r"`[^`]*/([^/`]+)`", r"`\1`", value)


def _outcome(value: str, sentences: int = 2) -> str:
    return " ".join(re.split(r"(?<=\.)\s+", value.strip())[:sentences])


def _run_key(record: pathlib.Path) -> tuple[str, list[int | str]]:
    # Folders are r<N> up to r47, then guparpit-<gym>-v<N> (user rule 2026-09-30).
    # Date first; digit runs compare as numbers (r41b before r41, v10 before v9).
    name = [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", record.parent.name)]
    return header(record).get("Date", ""), name


def rows(root: pathlib.Path) -> list[str]:
    """Return one Markdown table row per RECORD.md, newest run first within a family."""
    out = []
    # Newest run first (r41b, the r41 relaunch, before r41), then a stable sort by family.
    records = sorted(sorted(root.glob("*/*/RECORD.md"), key=_run_key, reverse=True), key=lambda r: r.parts[-3])
    for record in records:
        h = header(record)
        cells = (
            record.parent.name,
            h.get("Family", f"`{record.parts[-3]}`"),
            h.get("Date", "?"),
            h.get("Dataset", "?"),
            f"{_image(h.get('Gym image', '?'))}; {_image(h.get('Trainer image', '?'))}",
            h.get("Status", "?"),
            _outcome(h.get("Outcome", "?")),
            f"[{record.parent.name}/RECORD.md]({record.relative_to(root).as_posix()})",
        )
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return out


def render(root: pathlib.Path) -> str:
    """Return the INDEX.md text with a fresh Runs table between the markers."""
    text = (root / "INDEX.md").read_text()
    head, _, rest = text.partition(BEGIN)
    _, end, tail = rest.partition(END)
    if not end:
        sys.exit(f"FAIL: {root / 'INDEX.md'} has no {BEGIN} ... {END} block")
    table = "\n".join(["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |", *rows(root)])
    return f"{head}{BEGIN}\n{table}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    """Rewrite INDEX.md, or with --check report whether it is current."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=HERE, help="the training-runs/ directory")
    ap.add_argument("--check", action="store_true", help="exit 1 when the Runs table is stale; write nothing")
    a = ap.parse_args(argv)
    index = a.root / "INDEX.md"
    new = render(a.root)
    if a.check:
        return 0 if new == index.read_text() else 1
    index.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
