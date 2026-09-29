"""Write the kdatp-prof arm YAMLs from the ``kdatp`` T2 arm YAML of ``gen_arm_configs.py``.

    python3 patch_prof_arm.py <kdatp.yaml> <run-dir> --data-root <kdatp>/data [--arms-file <arms.yaml>] \
        [--groups 239,258] [--timeout 12600] [--skip-actor-forward-only]

It writes ``<run-dir>/arms/<name>.yaml`` for each arm and
``<run-dir>/arms/plan.tsv`` for the driver (one line per arm: name, groups,
dp_size, data dir, timeout, profile), and prints the keys that each arm
changes. The arms file is a YAML list (README.md). Without it, the plan is
one arm, ``kdatp-prof``: profiled, ``--groups``, DP 2.

Each arm keeps every key of the ``kdatp`` T2 arm (the r47 trainer layout),
then takes the data file and the batch shape of its groups, then the profiler
keys if ``profile`` is true, then its ``set`` dict, so ``set`` overrides any
key. The profiler runs on the ``train_overall`` target: step 0 (rollout 40)
warms up, step 1 (rollout 41) is profiled, step 2 (rollout 42) runs clean to
measure the profiler overhead. ``debug_exit_after_rollout`` 3 stops the run
after rollout 42. ``num_rollout`` stays 300: the train loop force-saves at
``num_rollout - 1``, so a smaller value adds a checkpoint save.
"""

import argparse
import re
from pathlib import Path

import yaml

# TrainProfiler (miles/utils/profile_utils.py): schedule wait = start - 1, warmup 1,
# active = end - start, one repeat. Traces go to tensorboard_dir as
# train_overall_rank_<rank>.<ns>.pt.trace.json.gz on every rank.
PROFILE = {
    "use_pytorch_profiler": True,
    "profile_target": ["train_overall"],
    "profile_step_start": 1,
    "profile_step_end": 2,
    "debug_exit_after_rollout": 3,
}
# The T2 arm sets these; the check below stops a base config that does not.
TRAIN_ONLY = {"num_rollout": 300, "save_interval": 100000, "use_wandb": False}
# The driver builds each data dir with these ids: rollout_40.pt, and 41..43 link to it.
# The run resumes at rollout 40 (the r43 iter_0000039 seed) and the train loop
# prefetches the next id, so an arm without replay_rollout trains at most 3 steps.
ROLLOUT_IDS = (40, 41, 42, 43)
GPUS_PER_NODE = 8
ARM_KEYS = {"name", "profile", "skip_actor_forward_only", "set", "data", "timeout"}
DATA_KEYS = {"groups", "dp_size", "replay_rollout", "dir"}
NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
GROUPS = re.compile(r"\d+(,\d+)*")
NOT_ARMS = {"arms", "a2a", "harness"}  # run-dir entries of the driver


def plan_arms(arms: object, groups: str, timeout: int, skip: bool) -> list[dict]:
    """Check each arm of the arms file and fill in the defaults."""
    assert isinstance(arms, list) and arms, "the arms file must hold a non-empty YAML list"
    plan = []
    for arm in arms:
        assert isinstance(arm, dict) and {"name", "profile"} <= set(arm), f"an arm needs name and profile: {arm!r}"
        assert set(arm) <= ARM_KEYS, f"arm {arm['name']!r}: unknown keys {sorted(set(arm) - ARM_KEYS)}"
        name = arm["name"]
        assert NAME.fullmatch(name) and name not in NOT_ARMS, f"bad arm name {name!r}"
        data = {"groups": groups, "dp_size": 2, "replay_rollout": None, **(arm.get("data") or {})}
        assert set(data) <= DATA_KEYS, f"arm {name!r}: unknown data keys {sorted(set(data) - DATA_KEYS)}"
        data["groups"] = str(data["groups"])  # YAML reads one group without quotes as an int
        assert GROUPS.fullmatch(data["groups"]), f"arm {name!r}: groups {data['groups']!r}"
        replay = data["replay_rollout"]
        assert replay is None or replay in ROLLOUT_IDS, f"arm {name!r}: replay_rollout {replay!r} not in {ROLLOUT_IDS}"
        # DP 2 keeps the name of the existing slice dirs (data/prof-239-258).
        suffix = "" if data["dp_size"] == 2 else f"-dp{data['dp_size']}"
        data.setdefault("dir", f"prof-{data['groups'].replace(',', '-')}{suffix}")
        assert NAME.fullmatch(data["dir"]), f"arm {name!r}: data dir {data['dir']!r} is not a name under data/"
        set_keys = arm.get("set") or {}
        assert isinstance(set_keys, dict), f"arm {name!r}: set must be a dict"
        skip_forward = arm.get("skip_actor_forward_only", skip)
        assert type(arm["profile"]) is bool and type(skip_forward) is bool, f"arm {name!r}: not a bool"
        plan.append(
            {
                "name": name,
                "profile": arm["profile"],
                "skip_actor_forward_only": skip_forward,
                "set": set_keys,
                "data": data,
                "timeout": int(arm.get("timeout", timeout)),
            }
        )
    names = [arm["name"] for arm in plan]
    assert len(set(names)) == len(names), f"duplicate arm names: {names}"
    return plan


def arm_config(base: dict, arm: dict, run_dir: str, data_root: str) -> dict:
    """The ``kdatp`` T2 arm plus the harness keys of ``arm``, then its ``set`` dict."""
    data = arm["data"]
    config = dict(base)
    # A path without {rollout_id} loads the same file at every step (debug_data.py str.format).
    rollout = "{rollout_id}" if data["replay_rollout"] is None else data["replay_rollout"]
    config["load_debug_rollout_data"] = f"{data_root}/{data['dir']}/rollout_{rollout}.pt"
    # One optimizer step: rollout_batch_size = groups, global_batch_size = groups x n_samples_per_prompt.
    num_groups = len(data["groups"].split(","))
    config["rollout_batch_size"] = num_groups
    config["global_batch_size"] = num_groups * config["n_samples_per_prompt"]
    config["experiment_name"] = "kdatp-prof"  # a record only; the pod env names the run
    config["project_name"] = "kdatp-prof"
    config["arena_sample_summary_dir"] = f"{run_dir}/{arm['name']}/sample_summary"
    if arm["profile"]:
        config.update(PROFILE)
        config["tensorboard_dir"] = f"{run_dir}/{arm['name']}/tb"
    if arm["skip_actor_forward_only"]:
        config["skip_actor_forward_only"] = True
    config.update(arm["set"])
    for key, value in TRAIN_ONLY.items():
        assert config.get(key) == value, f"arm {arm['name']!r}: set changes {key}; the arm must stay train only"
    if data["replay_rollout"] is None:
        steps = config["debug_exit_after_rollout"]
        assert steps < len(ROLLOUT_IDS), f"arm {arm['name']!r}: {steps} steps need replay_rollout"
    # build_rollout_data.py pads the rows to a multiple of the actor DP (gpus / (tp * pp * cp)).
    model_parallel = 1
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        model_parallel *= config.get(key) or 1
    dp = config["num_trainers"] * GPUS_PER_NODE // model_parallel
    assert dp == data["dp_size"], f"arm {arm['name']!r}: actor DP {dp}, data dp_size {data['dp_size']}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="the kdatp T2 arm of gen_arm_configs.py")
    parser.add_argument("run_dir")
    parser.add_argument("--data-root", required=True, help="<kdatp>/data")
    parser.add_argument("--arms-file", default=None)
    parser.add_argument("--groups", default="239,258", help="default groups")
    parser.add_argument("--timeout", type=int, default=12600, help="default arm timeout (s)")
    parser.add_argument(
        "--skip-actor-forward-only",
        action="store_true",
        help="default: r47 flip A, reuse the train-forward log probs, no separate log-prob pass",
    )
    cli = parser.parse_args()
    base = yaml.safe_load(Path(cli.src).read_text())
    for key, value in TRAIN_ONLY.items():
        assert base.get(key) == value, f"{cli.src}: {key}={base.get(key)!r}, expected {value!r}"
    default = [{"name": "kdatp-prof", "profile": True}]
    arms = yaml.safe_load(Path(cli.arms_file).read_text()) if cli.arms_file else default
    plan = plan_arms(arms, cli.groups, cli.timeout, cli.skip_actor_forward_only)
    out = Path(cli.run_dir) / "arms"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in plan:
        config = arm_config(base, arm, cli.run_dir, cli.data_root)
        (out / f"{arm['name']}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        data = arm["data"]
        fields = (arm["name"], data["groups"], data["dp_size"], data["dir"], arm["timeout"], int(arm["profile"]))
        rows.append("\t".join(map(str, fields)))
        print(f"arm {arm['name']}:")
        for key in sorted(set(config) | set(base)):
            if base.get(key) != config.get(key):
                print(f"  {key}: {base.get(key)!r} -> {config.get(key)!r}")
    (out / "plan.tsv").write_text("".join(row + "\n" for row in rows))
    print(f"{len(plan)} arms, timeout sum {sum(arm['timeout'] for arm in plan)} s")


if __name__ == "__main__":
    main()
