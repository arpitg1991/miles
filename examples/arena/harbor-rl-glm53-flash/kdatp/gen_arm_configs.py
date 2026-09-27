"""Write the train-only arm configs of the kdatp tests from the r45 config.

Each arm is the r45 ``miles-config.yaml`` with the common deltas of its test
and the arm deltas below. Every other r45 key stays the same, so a timing
difference between two arms comes from the arm deltas only.

    python3 gen_arm_configs.py t1c --data-dir <kdatp>/data --out <dir>   # 1 node, 5-layer slice
    python3 gen_arm_configs.py t2 --data-dir <kdatp>/data --out <dir>    # 8 nodes, the r45 layout

Arms:

- ``baseline``: today (all KDA heads on each rank, full recompute).
- ``kdatp``: item 3, ``glm5_next_kda_tp``.
- ``*-selective``: item 4, selective recompute of the Megatron modules that
  reach GLM-5.3 (``mhc``, ``moe_act``, ``layernorm``). ``mhc`` needs
  ``enable_hyper_connections`` on the command line, because Megatron checks
  it before the glm5_next spec turns mHC on.
- ``*-none``: no recompute (the T1c memory ceiling).
- ``kdatp-block10``: item 4 on top of item 3, full recompute of the first
  10 layers of each stage only.
"""

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
R45_CONFIG = HERE.parent / "r45" / "miles-config.yaml"

KDA_TP = {"glm5_next_kda_tp": True}
SELECTIVE = {
    "recompute_granularity": "selective",
    "recompute_method": None,
    "recompute_num_layers": None,
    "recompute_modules": ["mhc", "moe_act", "layernorm"],
    "enable_hyper_connections": True,
}
NO_RECOMPUTE = {"recompute_granularity": None, "recompute_method": None, "recompute_num_layers": None}
BLOCK10 = {"recompute_method": "block", "recompute_num_layers": 10}

# Keys that make the run train only, never save, and never log to a live W&B project.
TRAIN_ONLY = {
    "save_interval": 100000,  # the launcher always appends --save; no step reaches this
    "num_rollout": 300,
    "use_wandb": False,
}

TESTS = {
    "t1c": {
        "common": {
            "replicas": 1,
            "num_trainers": 1,
            "model_arch": "glm5.3-flash-5layer",
            "pipeline_model_parallel_size": 1,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "expert_model_parallel_size": 8,
            "rollout_batch_size": 1,
            "global_batch_size": 8,
            # No tracker in the fresh ckpt dir: miles loads ref_load (the base DCP) with finetune, rollout 0.
            "debug_exit_after_rollout": 1,
            "load_debug_rollout_data": "{data}/t1c/rollout_{{rollout_id}}.pt",
            # Config and tokenizer of the 5-layer slice (make_slice_hf.py); the weights come from ref_load.
            "hf_checkpoint": "{data}/hf/GLM-5.3-Flash-5layer",
        },
        "arms": {
            "baseline": {},
            "kdatp": KDA_TP,
            "selective": SELECTIVE,
            "kdatp-selective": {**KDA_TP, **SELECTIVE},
            "none": NO_RECOMPUTE,
            "kdatp-none": {**KDA_TP, **NO_RECOMPUTE},
        },
    },
    "t2": {
        "common": {
            "replicas": 8,
            "num_trainers": 8,
            "rollout_batch_size": 8,  # 8 groups x 8 = 64 episodes = one optimizer step
            "global_batch_size": 64,
            # Seeded from r43 iter_0000039: rollouts 40 to 43 (step 1 warms up, steps 2-4 time).
            "debug_exit_after_rollout": 4,
            "load_debug_rollout_data": "{data}/t2/rollout_{{rollout_id}}.pt",
        },
        "arms": {
            "baseline": {},
            "kdatp": KDA_TP,
            "kdatp-block10": {**KDA_TP, **BLOCK10},
            # The noise floor of the step-1 parity, and one more timed step.
            "baseline2": {"debug_exit_after_rollout": 2},
            # Last: the kdatp design predicts an out-of-memory error on stage 0 at 131K tokens.
            "kdatp-selective": {**KDA_TP, **SELECTIVE},
        },
    },
}


def arm_config(base: dict, test: str, arm: str, data_dir: str, out_dir: str) -> dict:
    spec = TESTS[test]
    config = {**base, **TRAIN_ONLY, **spec["common"], **spec["arms"][arm]}
    for key in ("load_debug_rollout_data", "hf_checkpoint"):
        config[key] = config[key].format(data=data_dir)
    config["experiment_name"] = f"kdatp-{test}-{arm}"  # a record only; the pod env names the run
    config["project_name"] = f"kdatp-{test}"
    config["arena_sample_summary_dir"] = f"{out_dir}/{arm}/sample_summary"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=str(R45_CONFIG))
    cli = parser.parse_args()
    base = yaml.safe_load(Path(cli.base).read_text())
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in TESTS[cli.test]["arms"]:
        path = out / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(arm_config(base, cli.test, arm, cli.data_dir, cli.out), sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
