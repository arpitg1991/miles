"""scripts/run_arena_harbor.py: the per-save HF export is opt-in (plugin ADR-0006 amendment 2026-09-27).

The default argv carries --load/--save and no --save-hf, so each save is DCP only.
``arena_save_hf: true`` in the YAML or a set ARENA_EVAL_TASKS appends
``--save-hf <ckpt>/hf/rollout_{rollout_id}``; the launcher key never reaches the trainer.
"""

import shlex

import pytest
import typer

from tests.fast.launch_scripts.py_harness import REPO_ROOT, import_launch_script

run = import_launch_script(REPO_ROOT / "scripts" / "run_arena_harbor.py")

_CKPT = "/ckpt/slime_experiments/rl-test"


def _argv(
    tmp_path, yaml_extra: str = "", arena_eval_tasks: str = "", replicas: int = 3, num_trainers: int = 2
) -> list[str]:
    config = tmp_path / "config.yaml"
    config.write_text(f"model_arch: qwen3.5-27B\nrollout_batch_size: 4\n{yaml_extra}")
    args = run.ScriptArgs(
        config=str(config),
        replicas=replicas,
        num_trainers=num_trainers,
        num_gpus_per_node=8,
        experiment_name="rl-test",
        checkpoints_dir="/ckpt",
        wandb_run_id="",
        arena_eval_tasks=arena_eval_tasks,
        use_mlflow_amzn=False,
    )
    _, train_args = run._build_train_args(args)
    return shlex.split(train_args)


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


@pytest.mark.parametrize("yaml_extra", ["", "arena_save_hf: false\n"])
def test_default_saves_dcp_only(tmp_path, yaml_extra):
    argv = _argv(tmp_path, yaml_extra)
    assert _value(argv, "--load") == _CKPT
    assert _value(argv, "--save") == _CKPT
    assert "--save-hf" not in argv
    assert "--arena-save-hf" not in argv


@pytest.mark.parametrize(
    ("yaml_extra", "arena_eval_tasks"),
    [("arena_save_hf: true\n", ""), ("arena-save-hf: true\n", ""), ("", "amzn_arena_tasks/some_task")],
)
def test_opt_in_appends_save_hf(tmp_path, yaml_extra, arena_eval_tasks):
    argv = _argv(tmp_path, yaml_extra, arena_eval_tasks)
    assert argv.count("--save-hf") == 1
    assert _value(argv, "--save-hf") == f"{_CKPT}/hf/rollout_{{rollout_id}}"
    assert _value(argv, "--save") == _CKPT
    assert "--arena-save-hf" not in argv


def test_non_boolean_value_fails_fast(tmp_path):
    with pytest.raises(typer.BadParameter, match="arena_save_hf"):
        _argv(tmp_path, 'arena_save_hf: "yes"\n')


_TRAIN_ONLY = "load_debug_rollout_data: /data/rollout_{rollout_id}.pt\n"


@pytest.mark.parametrize(("replicas", "rollout_gpus"), [(2, "8"), (3, "8")])
def test_train_only_runs_without_rollout_nodes(tmp_path, replicas, rollout_gpus):
    """debug_train_only places no rollout GPUs, so a trainer-only job gets one phantom node."""
    argv = _argv(tmp_path, _TRAIN_ONLY, replicas=replicas, num_trainers=2)
    assert _value(argv, "--rollout-num-gpus") == rollout_gpus
    assert _value(argv, "--actor-num-nodes") == "2"


def test_rollout_job_still_needs_a_rollout_node(tmp_path):
    with pytest.raises(AssertionError, match="rollout node"):
        _argv(tmp_path, replicas=2, num_trainers=2)
