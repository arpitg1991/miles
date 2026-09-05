"""Arena Harbor RL launcher: NATS-gym RL training on a kubeflow PyTorchJob.

=====================

Ports AGISlime's ``scripts/custom/entrypoint.sh`` + ``hydra_runner/hydra_converter.py``
to miles conventions for the harbor-rl jobs (e.g. Qwen3.5-27B + financeagent gym
workers, ``examples/arena/harbor-rl-27b/``). Every replica of the PyTorchJob runs this
script; the pod command picks the role from the kubeflow replica index:

  * ``worker`` (replicas 1..N-1): join the head's ray cluster and block. Never submits.
  * ``train`` (replica 0): start the ray head via ``execute_train``'s preamble, wait
    until all ``--replicas`` nodes are active, convert the experiment YAML into a flat
    argv, append the run-identity flags derived from the pod env, and submit
    ``miles_plugins/arena/train_async_arena.py``.

The YAML -> argv conversion keeps hydra_converter semantics exactly: a true boolean
becomes a bare flag, false/None keys emit nothing, lists become one flag followed by
one token per item, scalars are ``str()``'d (OmegaConf parses ``2e-6`` as a float, so
it is re-emitted as ``2e-06``), and ``prompt-data-list`` is renamed to
``--prompt-data`` and moved to the end.

Keys the launcher consumes from the YAML (miles argparse is strict, so unlike
slime 0.3.0 they must never reach the train argv): ``user``, ``cluster``,
``experiment_name``, ``project_name``, ``agislime_dir``, ``replicas``,
``num_trainers``, ``model_arch``. Only ``model_arch`` has a consumer here — it selects
``scripts/models/<model_arch>.py``. Run identity (checkpoint dir, wandb group) comes
from the ``EXPERIMENT_NAME`` pod env per ADR-0004, never from the YAML names; an
existing checkpoint dir resumes silently, so a finished run trains nothing and
exits 0 — bump ``EXPERIMENT_NAME`` for a fresh run.

=====================

Args: every ScriptArgs field is a flag; the defaults follow the pod env contract
  (REPLICA, REPLICA_TRAINER, GPUS_PER_NODE, CFG_NAME, EXPERIMENT_NAME, PROJECT_NAME,
  ARENA_CHECKPOINTS_DIR, ARENA_DATA_DIR, NATS_URL, ARENA_DEFAULT_GYM, WANDB_RUN_ID,
  ARENA_EVAL_TASKS, S3_ARTIFACT_BASE, USE_MLFLOW_AMZN). ``--rank`` and ``--head-addr``
  default from the kubeflow ``<job>-worker-<idx>`` hostname convention (REPLICA_IDX
  env first) and can be overridden on the CLI.

=====================

  on replicas 1..N-1:
    python scripts/run_arena_harbor.py worker
  on replica 0:
    python scripts/run_arena_harbor.py train
"""

import os
import re
import shlex
from dataclasses import dataclass, field

import typer
from omegaconf import OmegaConf

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_CONFIG = "examples/arena/harbor-rl-27b/financeagent-27b-smoke.yaml"

# YAML keys addressed to the launcher, not the trainer. slime 0.3.0 let them ride
# through argv unparsed (megatron ignore_unknown_args=True); miles hard-fails on
# unknown flags, so they are popped before conversion.
_LAUNCHER_CONSUMED_KEYS = (
    "user",
    "cluster",
    "experiment_name",
    "project_name",
    "agislime_dir",
    "replicas",
    "num_trainers",
    "model_arch",
)


def _default_rank() -> int:
    """Kubeflow replica index: REPLICA_IDX fieldRef env, else the hostname suffix."""
    replica_idx = os.environ.get("REPLICA_IDX", "")
    if replica_idx.isdigit():
        return int(replica_idx)
    match = re.fullmatch(r".+-worker-(\d+)", os.environ.get("HOSTNAME", ""))
    return int(match.group(1)) if match else 0


def _default_head_addr() -> str:
    """Replica 0's address from the kubeflow ``<job>-worker-<idx>`` hostname convention."""
    match = re.fullmatch(r"(.+-worker)-\d+", os.environ.get("HOSTNAME", ""))
    if match:
        return f"{match.group(1)}-0"
    return os.environ.get("MASTER_ADDR", "127.0.0.1")


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # experiment config; a relative path resolves under the miles checkout
    config: str = field(default_factory=lambda: os.environ.get("CFG_NAME", _DEFAULT_CONFIG))
    # cluster shape: replicas = trainer nodes + rollout nodes (pod env REPLICA/REPLICA_TRAINER)
    replicas: int = field(default_factory=lambda: int(os.environ.get("REPLICA", "3")))
    num_trainers: int = field(default_factory=lambda: int(os.environ.get("REPLICA_TRAINER", "2")))
    num_gpus_per_node: int = field(default_factory=lambda: int(os.environ.get("GPUS_PER_NODE", "8")))
    # run identity (ADR-0004): EXPERIMENT_NAME keys the checkpoint dir and the wandb group
    experiment_name: str = field(default_factory=lambda: os.environ.get("EXPERIMENT_NAME", "arena-smoke"))
    project_name: str = field(default_factory=lambda: os.environ.get("PROJECT_NAME", "arena-smoke"))
    checkpoints_dir: str = field(
        default_factory=lambda: os.environ.get("ARENA_CHECKPOINTS_DIR", "/root/shared_data/arena/checkpoints")
    )
    data_dir: str = field(default_factory=lambda: os.environ.get("ARENA_DATA_DIR", "/root/shared_data/arena/data"))
    # NATS wiring for the arena rollout workers; forwarded into the ray runtime env when set
    nats_url: str = field(default_factory=lambda: os.environ.get("NATS_URL", ""))
    default_gym: str = field(default_factory=lambda: os.environ.get("ARENA_DEFAULT_GYM", ""))
    # overrides the YAML `model_arch` key (-> scripts/models/<model_arch>.py)
    model_arch: str = ""
    head_addr: str = field(default_factory=_default_head_addr)
    rank: int = field(default_factory=_default_rank)
    wandb_run_id: str = field(default_factory=lambda: os.environ.get("WANDB_RUN_ID", ""))
    arena_eval_tasks: str = field(default_factory=lambda: os.environ.get("ARENA_EVAL_TASKS", ""))
    s3_artifact_base: str = field(default_factory=lambda: os.environ.get("S3_ARTIFACT_BASE", ""))
    s3_artifact_region: str = field(default_factory=lambda: os.environ.get("S3_ARTIFACT_REGION", "ap-south-1"))
    # detected only to fail fast: mlflow-amzn is not supported on the miles arena path
    use_mlflow_amzn: bool = field(default_factory=lambda: os.environ.get("USE_MLFLOW_AMZN", "false") == "true")
    run_id: str = field(default_factory=lambda: U.create_run_id())
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    @property
    def config_path(self) -> str:
        return self.config if os.path.isabs(self.config) else f"{U.repo_base_dir}/{self.config}"

    @property
    def ckpt_dir(self) -> str:
        """DCP --load/--save dir; must stay on a POSIX-faithful mount (ADR-0004)."""
        return f"{self.checkpoints_dir}/slime_experiments/{self.experiment_name}"

    @property
    def hf_save_dir(self) -> str:
        """--save-hf destination; ``{rollout_id}`` is substituted by the save-hf path itself.

        Always the local ckpt-dir form, even when S3_ARTIFACT_BASE is set:
        miles' save_hf_model consumes --save-hf strictly as a local Path, so the
        s3:// URI entrypoint.sh emitted here was mangled into a pod-local
        's3:/…' directory and never uploaded. Trainer-side HF exports stay
        local; the S3 upload of eval checkpoints happens in eval_rollout via
        s3_artifact (which is why S3_ARTIFACT_BASE still exports the AWS region
        into the runtime env below).
        """
        return f"{self.ckpt_dir}/hf/rollout_{{rollout_id}}"

    @property
    def log_dir(self) -> str:
        return f"{self.data_dir}/logs"

    @property
    def tb_dir(self) -> str:
        return f"{self.data_dir}/tb_logs/{self.project_name}/{self.experiment_name}_{self.run_id}"

    @property
    def wandb_dir(self) -> str:
        # per-experiment wandb dir; without it wandb drops metadata under cwd
        return f"{self.log_dir}/wandb/{self.project_name}/{self.experiment_name}"


def _flatten(prefix: str, obj: object, out: list[str]) -> None:
    """hydra_converter.py's flatten, verbatim: keys -> --hyphenated flags, bool true ->
    bare flag, bool false / None -> dropped, list -> one flag + one token per item."""
    flag = f"--{prefix.replace('_', '-')}"
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}-{k}" if prefix else k
            _flatten(key, v, out)
    elif isinstance(obj, list):
        out.append(flag)
        for item in obj:
            out.append(str(item))
    elif isinstance(obj, bool):
        if obj:
            out.append(flag)
    elif obj is not None:
        out.append(flag)
        out.append(str(obj))


def _load_experiment_config(config_path: str) -> dict:
    cfg = OmegaConf.load(config_path)
    # resolve all interpolations (including included defaults)
    cfg = OmegaConf.merge(cfg)
    container = OmegaConf.to_container(cfg, resolve=True)
    # remap gym-evals keys to the trainer argparser names; OmegaConf preserves
    # hyphens from YAML keys, and the pop/re-insert moves prompt-data last
    for key in ("prompt-data-list", "prompt_data_list"):
        if key in container:
            container["prompt-data"] = container.pop(key)
            break
    return container


def _pop_launcher_keys(container: dict) -> dict:
    consumed = {}
    for key in _LAUNCHER_CONSUMED_KEYS:
        for variant in (key, key.replace("_", "-")):
            if variant in container:
                consumed[key] = container.pop(variant)
    return consumed


def _build_train_args(args: ScriptArgs) -> tuple[str, str]:
    """Returns (megatron_model_type, train argv string) for execute_train."""
    container = _load_experiment_config(args.config_path)
    consumed = _pop_launcher_keys(container)
    tokens: list[str] = []
    _flatten("", container, tokens)

    # entrypoint.sh appends, same conditions: wandb-group tracks the deployed
    # EXPERIMENT_NAME so each deploy gets a distinct W&B run name
    if args.experiment_name:
        tokens += ["--wandb-group", args.experiment_name]
    # the deploy-generated run-id lands trainer / gym workers / eval in one wandb run
    if args.wandb_run_id and "--wandb-run-id" not in tokens:
        tokens += ["--wandb-run-id", args.wandb_run_id]
    # per-checkpoint eval-on-save (opt-in via ARENA_EVAL_TASKS) resolves the run by
    # display name, so pin it to --wandb-group
    if args.arena_eval_tasks and "--disable-wandb-random-suffix" not in tokens:
        tokens += ["--disable-wandb-random-suffix"]

    rollout_nodes = args.replicas - args.num_trainers
    assert rollout_nodes > 0, f"need at least one rollout node: replicas={args.replicas} trainers={args.num_trainers}"
    tokens += [
        "--actor-num-nodes", str(args.num_trainers),
        "--actor-num-gpus-per-node", str(args.num_gpus_per_node),
        "--num-gpus-per-node", str(args.num_gpus_per_node),
        "--rollout-num-gpus", str(rollout_nodes * args.num_gpus_per_node),
        # --load == --save: an existing checkpoint here resumes silently (ADR-0004)
        "--load", args.ckpt_dir,
        "--save", args.ckpt_dir,
        "--save-hf", args.hf_save_dir,
    ]  # fmt: skip
    if args.use_mlflow_amzn:
        # entrypoint.sh appended `--use-mlflow-amzn --mlflow-project <p>
        # --mlflow-group <g>` here. No consumer registers those flags in this tree
        # (they were silently-dropped no-ops on slime 0.3.0 too), so on miles'
        # strict parser they would kill the trainer at argparse deep inside the
        # ray job. Fail fast at the launcher instead. Both 27B smoke jobs leave
        # USE_MLFLOW_AMZN unset.
        raise typer.BadParameter(
            "USE_MLFLOW_AMZN=true: mlflow-amzn is not supported on the miles arena "
            "path yet — no parser registers --use-mlflow-amzn/--mlflow-project/"
            "--mlflow-group. Unset USE_MLFLOW_AMZN (miles' native tracking flags "
            "are --use-mlflow/--mlflow-tracking-uri/--mlflow-experiment-name)."
        )

    model_arch = args.model_arch or str(consumed.get("model_arch") or "")
    assert model_arch, "model_arch must be set via --model-arch or the experiment YAML"
    # quoting keeps the --prompt-data JSON (literal double quotes, [{...}] glob
    # characters) intact through the bash -c the command goes through
    train_args = " ".join(shlex.quote(token) for token in tokens)
    if args.extra_args:
        train_args = f"{train_args} {args.extra_args}"
    return model_arch, train_args


def _wait_for_ray_nodes(head_addr: str, num_nodes: int) -> None:
    """entrypoint.sh's wait-for-all-workers loop, bounded: count Active nodes in ray
    status; after 10 minutes submit anyway so the job's own error names the shortfall."""
    count_cmd = (
        f"ray status --address={head_addr}:6379 2>/dev/null "
        "| sed -n '/^Active:/,/^Pending:/p' | grep -c 'node_' || true"
    )
    for _ in range(120):
        active = (U.exec_command_cpu(count_cmd, capture_output=True) or "").strip() or "0"
        if int(active) >= num_nodes:
            print(f"[ray] cluster ready: {active}/{num_nodes} active nodes")
            break
        print(f"waiting for all workers up... ({active}/{num_nodes} active)")
        U.exec_command_cpu("sleep 5")
    U.exec_command_cpu(f"ray status --address={head_addr}:6379")


def _wait_for_head_port(head_addr: str) -> None:
    for _ in range(120):
        # exec_command_cpu raises on a non-zero exit, so the probe reports through stdout.
        if U.exec_command_cpu(f"nc -z {head_addr} 6379 2>/dev/null; echo $?", capture_output=True).strip() == "0":
            return
        print(f"waiting for ray head {head_addr}:6379 ...")
        U.exec_command_cpu("sleep 5")


def _execute_train(args: ScriptArgs) -> None:
    megatron_model_type, train_args = _build_train_args(args)
    dirs = " ".join(shlex.quote(path) for path in (args.ckpt_dir, args.log_dir, args.tb_dir, args.wandb_dir))
    U.exec_command_cpu(f"mkdir -p {dirs}")

    extra_env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TENSORBOARD_DIR": args.tb_dir,
        "WANDB_DIR": args.wandb_dir,
        "RAY_enable_open_telemetry_metrics": "false",
        # the NATS rollout worker inside the RolloutManager actor reads these
        **({"NATS_URL": args.nats_url} if args.nats_url else {}),
        **({"ARENA_DEFAULT_GYM": args.default_gym} if args.default_gym else {}),
        # boto3 needs a region for the S3 artifact bucket: trainer-side HF exports
        # stay local (see hf_save_dir), but the eval path's own DCP->HF-convert-and
        # -upload flow (eval_rollout via s3_artifact) reads these on the ray actors
        **(
            {"AWS_REGION": args.s3_artifact_region, "AWS_DEFAULT_REGION": args.s3_artifact_region}
            if args.s3_artifact_base
            else {}
        ),
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script="miles_plugins/arena/train_async_arena.py",
        before_ray_job_submit=lambda: _wait_for_ray_nodes(args.head_addr, args.replicas),
        extra_env_vars=extra_env_vars,
        config=args,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Head role (replica 0): start the ray head, wait for all nodes, submit the job."""
    assert args.rank == 0, f"train must run on replica 0 (detected rank {args.rank}); use the worker command"
    # execute_train reads MASTER_ADDR for the ray head's node ip and for the torch
    # distributed rendezvous the worker ranks connect to.
    os.environ["MASTER_ADDR"] = args.head_addr
    _execute_train(args)


@app.command()
@U.dataclass_cli
def worker(args: ScriptArgs):
    """Worker role (replicas 1..N-1): join the head's ray cluster and block."""
    # A restarted pod inherits the previous run's agents, and ray refuses to join with them alive.
    U.exec_command_cpu("pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 miles; sleep 3; true; ")
    _wait_for_head_port(args.head_addr)
    U.exec_command_cpu(
        f"ray start --address={args.head_addr}:6379 "
        f"--num-gpus={args.num_gpus_per_node} "
        "--disable-usage-stats "
        "--block"
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
