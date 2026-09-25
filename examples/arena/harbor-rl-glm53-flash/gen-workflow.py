"""Emit r<N>/workflow.yaml for a guparpit-miles-deployer WorkflowTemplate.

Usage: .venv python gen-workflow.py <N> [--base r27] [--template ...]
       [--experiment-name NAME] [--gym-image IMAGE] [--param NAME=VALUE ...]

Copies every submit parameter from the base workflow.yaml except
partial-reward, then sets the run identity, the miles-config block
(r<N>/miles-config.yaml verbatim), the gym image (r<N>/gym-worker.yaml
unless --gym-image is given) and any --param. Checks that a re-parse of
the output returns the config byte for byte.
"""

import argparse
import pathlib
import re
import sys

import yaml


class _Block(str):
    """String that PyYAML emits as a literal block scalar."""


def _block_repr(dumper: yaml.Dumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_Block, _block_repr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--base", default="r19")
    ap.add_argument("--template", default="guparpit-miles-deployer-v4")
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
    a = ap.parse_args()
    here = pathlib.Path(__file__).parent
    run = here / f"r{a.n}"
    wf = yaml.safe_load((here / a.base / "workflow.yaml").read_text())
    cfg = (run / "miles-config.yaml").read_text()
    gym_img = a.gym_image or re.search(r"image: (\S+arena-slime-dev:gym-\S+)", (run / "gym-worker.yaml").read_text()).group(1)
    wf["metadata"]["generateName"] = f"rl-glm53f{a.n}-"
    # RBAC allows create but not patch on workflowtemplates, so each template
    # revision gets a new name.
    wf["spec"]["workflowTemplateRef"]["name"] = a.template
    # AREnATasks ADR-0069: the gym trains on the Harbor trial reward and the
    # template has no partial-reward parameter. r20..r38 still carry it.
    wf_params = wf["spec"]["arguments"]["parameters"]
    wf_params[:] = [p for p in wf_params if p["name"] != "partial-reward"]
    params = {p["name"]: p for p in wf_params}
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
        if name in params:
            params[name]["value"] = value
        else:
            wf_params.append({"name": name, "value": value})
    out.write_text(yaml.dump(wf, sort_keys=False, width=10**9))
    back = yaml.safe_load(out.read_text())
    got = {p["name"]: p["value"] for p in back["spec"]["arguments"]["parameters"]}
    if got["miles-config"] != cfg:
        sys.exit("miles-config round-trip mismatch")
    print(out, "generateName", back["metadata"]["generateName"], "trainer-image", got["trainer-image"], "gym-image", got["gym-image"], "experiment", got["experiment-name"])


if __name__ == "__main__":
    main()
