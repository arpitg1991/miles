"""Emit r<N>/workflow.yaml for the guparpit-miles-deployer-v2 WorkflowTemplate.

Usage: .venv python gen-workflow.py <N> [--partial-reward ctrf]

Copies every submit parameter from r19/workflow.yaml, then sets the run
identity, the miles-config block (r<N>/miles-config.yaml verbatim), the
gym image (r<N>/gym-worker.yaml) and the partial-reward mode. Checks that
a re-parse of the output returns the config byte for byte.
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
    ap.add_argument("--partial-reward", default="ctrf")
    ap.add_argument("--base", default="r19")
    ap.add_argument("--template", default="guparpit-miles-deployer-v2")
    a = ap.parse_args()
    here = pathlib.Path(__file__).parent
    run = here / f"r{a.n}"
    wf = yaml.safe_load((here / a.base / "workflow.yaml").read_text())
    cfg = (run / "miles-config.yaml").read_text()
    gym_img = re.search(r"image: (\S+arena-slime-dev:gym-\S+)", (run / "gym-worker.yaml").read_text()).group(1)
    wf["metadata"]["generateName"] = f"rl-glm53f{a.n}-"
    # RBAC allows create but not patch on workflowtemplates, so each template
    # revision gets a new name.
    wf["spec"]["workflowTemplateRef"]["name"] = a.template
    params = {p["name"]: p for p in wf["spec"]["arguments"]["parameters"]}
    params["experiment-name"]["value"] = f"rl-glm53f-gbash-r{a.n}"
    params["miles-config"]["value"] = _Block(cfg)
    # r19 left gym-image and partial-reward at the template defaults.
    for name, value in (("gym-image", gym_img), ("partial-reward", a.partial_reward)):
        if name in params:
            params[name]["value"] = value
        else:
            wf["spec"]["arguments"]["parameters"].append({"name": name, "value": value})
    out = run / "workflow.yaml"
    out.write_text(yaml.dump(wf, sort_keys=False, width=10**9))
    back = yaml.safe_load(out.read_text())
    got = {p["name"]: p["value"] for p in back["spec"]["arguments"]["parameters"]}
    if got["miles-config"] != cfg:
        sys.exit("miles-config round-trip mismatch")
    print(out, "generateName", back["metadata"]["generateName"], "gym-image", got["gym-image"], "partial-reward", got["partial-reward"], "experiment", got["experiment-name"])


if __name__ == "__main__":
    main()
