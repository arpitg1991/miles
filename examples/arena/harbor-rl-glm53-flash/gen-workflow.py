"""Emit r<N>/workflow.yaml and the r<N>/RECORD.md header for a guparpit-miles-deployer WorkflowTemplate.

Usage: .venv python gen-workflow.py <N> [--base r27] [--template ...]
       [--experiment-name NAME] [--gym-image IMAGE] [--param NAME=VALUE ...]
       [--runs-dir DIR]

The run folders live in training-runs/harbor-rl-glm53-flash/ (see
training-runs/README.md). This directory holds the recipe and the code.

Copies every submit parameter from the base workflow.yaml except
partial-reward and step-cut-on-fail. A base compaction-max N becomes
agent-kwargs {"max_compactions": N}. Then sets the run identity, the
miles-config block (r<N>/miles-config.yaml verbatim), the gym image
(r<N>/gym-worker.yaml unless --gym-image is given) and any --param. Checks
that a re-parse of the output returns the config byte for byte.

Then writes the header block of r<N>/RECORD.md, between the
`<!-- gen-workflow:begin -->` and `<!-- gen-workflow:end -->` markers of
training-runs/TEMPLATE.md: date, family, generateName, experiment name, W&B
project, dataset rows and lakeFS refs, gym and trainer images, template, and
base. Text outside the markers stays as it is. A config without a
prompt-data-list row is a hard FAIL, so no record can omit the dataset.
"""

import argparse
import datetime as dt
import json
import pathlib
import re
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
# examples/arena/<family>/ -> <repo>/training-runs/<family>/
RUNS_DIR = HERE.parents[2] / "training-runs" / HERE.name
RECORD_BEGIN = "<!-- gen-workflow:begin -->"
RECORD_END = "<!-- gen-workflow:end -->"
_HEADER_LINE = re.compile(r"^\*\*(.+?):\*\* (.*)$", re.M)
_LAKEFS_REF = re.compile(r"lakefs://[^/]+/([^/]+)/")


class _Block(str):
    """String that PyYAML emits as a literal block scalar."""


def _block_repr(dumper: yaml.Dumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_Block, _block_repr)


def dataset_fields(cfg: dict) -> tuple[str, str]:
    """Return the Dataset and Manifest commit header values of a parsed miles-config.

    The dataset is the `prompt-data-list` key, a JSON list of rows with `path`
    and `gym_name`. It is not a workflow parameter. A missing row is a FAIL.
    """
    raw = cfg.get("prompt-data-list")
    rows = json.loads(raw) if isinstance(raw, str) else (raw or [])
    if not rows or any(not row.get("path") for row in rows):
        sys.exit("FAIL: miles-config.yaml has no prompt-data-list row with a path; the run record needs the dataset")
    dataset = "; ".join(f"`{row['path']}` (gym `{row.get('gym_name', '?')}`)" for row in rows)
    refs = []
    for row in rows:
        m = _LAKEFS_REF.match(row["path"])
        refs.append(f"`{m.group(1)}`" if m else "none (not a lakeFS URI)")
    return dataset, "; ".join(refs)


def write_record(run: pathlib.Path, template: pathlib.Path, fields: dict[str, str], base: str) -> pathlib.Path:
    """Create r<N>/RECORD.md from the template, or refresh only its marker block."""
    record = run / "RECORD.md"
    if record.exists():
        text = record.read_text()
    else:
        text = template.read_text().replace("r<N>", run.name).replace("<N>", run.name[1:]).replace("r<M>", base)
    head, _, rest = text.partition(RECORD_BEGIN)
    old_block, end, tail = rest.partition(RECORD_END)
    if not end:
        sys.exit(f"FAIL: {record} has no {RECORD_BEGIN} ... {RECORD_END} block")
    old_date = dict(_HEADER_LINE.findall(old_block)).get("Date", "")
    # The date is the day the record was first written; a re-run keeps it.
    date = old_date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", old_date) else dt.datetime.now(dt.UTC).date().isoformat()
    block = "\n".join(f"**{key}:** {value}" for key, value in ({"Date": date} | fields).items())
    record.write_text(f"{head}{RECORD_BEGIN}\n{block}\n{RECORD_END}{tail}")
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--base", default="r19")
    ap.add_argument("--template", default="guparpit-miles-deployer-v4")
    ap.add_argument("--runs-dir", type=pathlib.Path, default=RUNS_DIR, help="training-runs/<family>/ with the r<N> folders")
    # Default: keep the trainer image of the existing r<N>/workflow.yaml. The
    # base workflow carries an older tag, so a plain regeneration silently
    # downgraded r26 (r6 -> r3) on 2026-09-17.
    ap.add_argument("--trainer-image", default=None)
    ap.add_argument("--experiment-name", default=None, help="default rl-glm53f-gbash-r<N>")
    # r28: the gym image is a placeholder until the image build lands, so the
    # gym-worker.yaml regex cannot supply it.
    ap.add_argument("--gym-image", default=None)
    # Template parameters the base workflow does not carry (r28: the v5
    # deadline knobs and the gym name). Appended when absent, replaced when present.
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    a = ap.parse_args(argv)
    runs = a.runs_dir
    run = runs / f"r{a.n}"
    wf = yaml.safe_load((runs / a.base / "workflow.yaml").read_text())
    cfg = (run / "miles-config.yaml").read_text()
    # The dataset check runs first, so a FAIL leaves no half-written run folder.
    dataset, manifest_commit = dataset_fields(yaml.safe_load(cfg) or {})
    gym_img = a.gym_image or re.search(r"image: (\S+arena-slime-dev:gym-\S+)", (run / "gym-worker.yaml").read_text()).group(1)
    wf["metadata"]["generateName"] = f"rl-glm53f{a.n}-"
    # RBAC allows create but not patch on workflowtemplates, so each template
    # revision gets a new name.
    wf["spec"]["workflowTemplateRef"]["name"] = a.template
    # AREnATasks ADR-0069: the gym trains on the Harbor trial reward and the
    # template has no partial-reward parameter. r20..r38 still carry it.
    wf_params = wf["spec"]["arguments"]["parameters"]
    # Apps ADR-0016 amendment 2026-09-27: the template has no compaction-max
    # parameter, and Argo keeps an undeclared argument without an error. r28..r43
    # carry it, so their value moves to agent-kwargs, the only Vulcan path.
    # Apps 658dfe4 removed step-cut-on-fail (v10 and later). r31..r43 carry it.
    compaction_max = next((p["value"] for p in wf_params if p["name"] == "compaction-max"), None)
    dropped = ("partial-reward", "compaction-max", "step-cut-on-fail")
    wf_params[:] = [p for p in wf_params if p["name"] not in dropped]
    params = {p["name"]: p for p in wf_params}
    if compaction_max is not None and "agent-kwargs" not in params:
        params["agent-kwargs"] = {
            "name": "agent-kwargs",
            "value": json.dumps({"max_compactions": int(compaction_max)}),
        }
        wf_params.append(params["agent-kwargs"])
    params["experiment-name"]["value"] = a.experiment_name or f"rl-glm53f-gbash-r{a.n}"
    out = run / "workflow.yaml"
    trainer_image = a.trainer_image
    if trainer_image is None and out.exists():
        prev = {p["name"]: p["value"] for p in yaml.safe_load(out.read_text())["spec"]["arguments"]["parameters"]}
        trainer_image = prev.get("trainer-image")
    if trainer_image is not None:
        params["trainer-image"]["value"] = trainer_image
    params["miles-config"]["value"] = _Block(cfg)
    # r19 left gym-image at the template default.
    extra = [tuple(kv.split("=", 1)) for kv in a.param]
    for name, value in (("gym-image", gym_img), *extra):
        if name == "compaction-max":
            sys.exit("compaction-max is gone: pass --param agent-kwargs='{\"max_compactions\": N}'")
        if name in params:
            params[name]["value"] = value
        else:
            wf_params.append({"name": name, "value": value})
    out.write_text(yaml.dump(wf, sort_keys=False, width=10**9))
    back = yaml.safe_load(out.read_text())
    got = {p["name"]: p["value"] for p in back["spec"]["arguments"]["parameters"]}
    if got["miles-config"] != cfg:
        sys.exit("miles-config round-trip mismatch")
    wandb_project = (yaml.safe_load(cfg) or {}).get("wandb_project", "not in config")
    fields = {
        "Family": f"`{runs.name}`",
        "Argo generateName": f"`{back['metadata']['generateName']}`",
        "Experiment name": f"`{got['experiment-name']}`",
        "W&B project": f"`{wandb_project}` (group `{got['experiment-name']}`)",
        "Dataset": dataset,
        "Manifest commit": manifest_commit,
        "Gym image": f"`{got['gym-image']}`",
        "Trainer image": f"`{got['trainer-image']}`",
        "Template": f"`{back['spec']['workflowTemplateRef']['name']}`",
        "Base": f"`{a.base}`",
    }
    record = write_record(run, runs.parent / "TEMPLATE.md", fields, a.base)
    print(out, "generateName", back["metadata"]["generateName"], "trainer-image", got["trainer-image"], "gym-image", got["gym-image"], "experiment", got["experiment-name"], "agent-kwargs", got.get("agent-kwargs"), "record", record)


if __name__ == "__main__":
    main()
