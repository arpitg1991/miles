"""scripts/run_arena_harbor.py: the per-save HF export is opt-in (plugin ADR-0006 amendment 2026-09-27).

The default argv carries --load/--save and no --save-hf, so each save is DCP only.
``arena_save_hf: true`` in the YAML or a set ARENA_EVAL_TASKS appends
``--save-hf <ckpt>/hf/rollout_{rollout_id}``; the launcher key never reaches the trainer.
An R3 launch needs an SGLang MoE runner that materializes the top-k ids.
"""

import shlex

import pytest
import typer

from tests.fast.launch_scripts.py_harness import REPO_ROOT, import_launch_script

run = import_launch_script(REPO_ROOT / "scripts" / "run_arena_harbor.py")

_CKPT = "/ckpt/slime_experiments/rl-test"


def _argv(
    tmp_path,
    yaml_extra: str = "",
    arena_eval_tasks: str = "",
    replicas: int = 3,
    num_trainers: int = 2,
    config: str | None = None,
    extra_args: str = "",
) -> list[str]:
    if config is None:
        config = str(tmp_path / "config.yaml")
        (tmp_path / "config.yaml").write_text(f"model_arch: qwen3.5-27B\nrollout_batch_size: 4\n{yaml_extra}")
    args = run.ScriptArgs(
        config=config,
        replicas=replicas,
        num_trainers=num_trainers,
        num_gpus_per_node=8,
        experiment_name="rl-test",
        checkpoints_dir="/ckpt",
        wandb_run_id="",
        arena_eval_tasks=arena_eval_tasks,
        use_mlflow_amzn=False,
        extra_args=extra_args,
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


@pytest.mark.parametrize(
    ("yaml_extra", "expected"),
    [
        ("", []),
        ("mini_ft_controller_enable: false\n", []),
        ("no_mini_ft_controller_enable: true\n", ["--no-mini-ft-controller-enable"]),
    ],
)
def test_yaml_false_emits_no_flag(tmp_path, yaml_extra, expected):
    """A YAML ``false`` is dropped, so it cannot turn off a flag that resolves to True.

    ``--mini-ft-controller-enable`` defaults to None, and ``--use-fault-tolerance``
    resolves it to True (``miles.utils.arguments``). Only the ``no_`` key reaches
    the parser as a negation.
    """
    argv = _argv(tmp_path, "use_fault_tolerance: true\n" + yaml_extra)
    assert [token for token in argv if "mini-ft-controller" in token] == expected


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


_R3 = "use_rollout_routing_replay: true\n"
_FAMILY = REPO_ROOT / "training-runs" / "harbor-rl-glm53-flash"


@pytest.mark.parametrize(
    ("yaml_extra", "extra_args"),
    [
        (_R3, ""),
        (_R3 + "sglang_moe_runner_backend: auto\n", ""),
        (_R3 + "sglang_moe_runner_backend: flashinfer_trtllm\n", ""),
        (_R3 + "sglang_moe_runner_backend: triton\n", "--sglang-moe-runner-backend auto"),
        (_R3 + "sglang_moe_runner_backend: flashinfer_cutlass\n", ""),
        (_R3 + "sglang_moe_runner_backend: triton\n", "--sglang-moe-runner-backend=auto"),
    ],
    ids=["unset", "auto", "flashinfer_trtllm", "extra_args_override", "unverified_runner", "equals_form_override"],
)
def test_r3_without_a_topk_moe_runner_fails_fast(tmp_path, yaml_extra, extra_args):
    """flashinfer_trtllm captures no routed experts, and sm100 resolves auto to it."""
    with pytest.raises(typer.BadParameter, match="materializes the top-k ids") as info:
        _argv(tmp_path, yaml_extra, extra_args=extra_args)
    assert "sglang_moe_runner_backend: triton" in str(info.value)
    assert "scripts/run_glm5_3_flash.py" in str(info.value)


@pytest.mark.parametrize(
    ("yaml_extra", "extra_args"),
    [
        (_R3 + "sglang_moe_runner_backend: triton\n", ""),
        (_R3, "--sglang-moe-runner-backend triton"),
        (_R3, "--sglang-moe-runner-backend=triton"),
        ("sglang_moe_runner_backend: auto\n", ""),
        ("use_rollout_routing_replay: false\nsglang_moe_runner_backend: auto\n", ""),
        # A train-only run starts no SGLang engine.
        (_R3 + _TRAIN_ONLY, ""),
    ],
    ids=["triton", "extra_args_triton", "equals_form_triton", "no_r3", "r3_false", "train_only"],
)
def test_r3_moe_runner_check_passes(tmp_path, yaml_extra, extra_args):
    _argv(tmp_path, yaml_extra, extra_args=extra_args)


def test_r3_moe_runner_check_on_the_agentic_debt_configs(tmp_path):
    """The v2 config sets triton. The r48 config ran on SGLang 9a26e749, which left flashinfer_trtllm by itself."""
    argv = _argv(tmp_path, config=str(_FAMILY / "guparpit-agentic-debt-v2" / "miles-config.yaml"), replicas=10)
    assert _value(argv, "--sglang-moe-runner-backend") == "triton"
    with pytest.raises(typer.BadParameter, match="unset"):
        _argv(tmp_path, config=str(_FAMILY / "guparpit-agentic-debt-v1" / "miles-config.yaml"), replicas=10)
