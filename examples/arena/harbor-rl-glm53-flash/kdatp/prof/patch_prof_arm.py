"""Write the ``kdatp-prof`` arm from the ``kdatp`` T2 arm YAML of ``gen_arm_configs.py``.

    python3 patch_prof_arm.py <kdatp.yaml> <kdatp-prof.yaml> --tensorboard-dir <dir> \
        [--sample-summary-dir <dir>] [--skip-actor-forward-only] [--set key=value ...]

The arm keeps every key of the ``kdatp`` T2 arm (the r47 trainer layout) and
adds the torch profiler on the ``train_overall`` target: step 0 (rollout 40)
warms up, step 1 (rollout 41) is profiled, step 2 (rollout 42) runs clean to
measure the profiler overhead. ``debug_exit_after_rollout`` 3 stops the run
after rollout 42. ``num_rollout`` stays 300: the train loop force-saves at
``num_rollout - 1``, so a smaller value adds a checkpoint save.

``--set key=value`` (YAML value) overrides any other key, for example the
data file and the batch shape of a smaller step: Kineto holds at most 128 MiB
of GPU activity records per trace, and torch 2.13 skips the record phase when
the warmup step alone fills that buffer (run 20260929a: a full T2 step did).
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--sample-summary-dir", default=None)
    parser.add_argument(
        "--skip-actor-forward-only",
        action="store_true",
        help="r47 flip A: reuse the train-forward log probs, no separate log-prob pass",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="YAML value")
    cli = parser.parse_args()
    config = yaml.safe_load(Path(cli.src).read_text())
    for key, value in TRAIN_ONLY.items():
        assert config.get(key) == value, f"{cli.src}: {key}={config.get(key)!r}, expected {value!r}"
    config.update(PROFILE)
    config["tensorboard_dir"] = cli.tensorboard_dir
    config["experiment_name"] = "kdatp-prof"  # a record only; the pod env names the run
    config["project_name"] = "kdatp-prof"
    if cli.sample_summary_dir:
        config["arena_sample_summary_dir"] = cli.sample_summary_dir
    if cli.skip_actor_forward_only:
        config["skip_actor_forward_only"] = True
    for item in cli.set:
        key, _, value = item.partition("=")
        assert key and _ == "=", f"--set expects KEY=VALUE, got {item!r}"
        config[key] = yaml.safe_load(value)
    Path(cli.dst).write_text(yaml.safe_dump(config, sort_keys=False))
    base = yaml.safe_load(Path(cli.src).read_text())
    for key in sorted(set(config) | set(base)):
        if base.get(key) != config.get(key):
            print(f"{key}: {base.get(key)!r} -> {config.get(key)!r}")


if __name__ == "__main__":
    main()
