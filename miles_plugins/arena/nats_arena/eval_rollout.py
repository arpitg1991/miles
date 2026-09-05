"""Fire-and-forget eval trigger for miles <-> AREnABase integration.

Miles calls ``generate_rollout(args, rollout_id, data_source, evaluation=True)``
at every ``--eval-interval`` step. This module:

  1. Resolves the HF checkpoint for this step.
  2. Renders an eval-coordinator-job.yaml (a K8s Job).
  3. ``kubectl apply``s it (or via the Python client) and returns IMMEDIATELY
     with empty data.

The coordinator pod handles everything else (deploys eval-sglang + eval-gym,
publishes tasks, collects results, logs to wandb, self-destructs). Multiple
concurrent coordinators can run for different eval steps.

Wired in via ``--eval-function-path miles_plugins.arena.nats_arena.eval_rollout.generate_rollout``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from miles.rollout.base_types import RolloutFnEvalOutput

logger = logging.getLogger(__name__)

__all__ = ["generate_rollout"]


_TEMPLATE_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _render(template: str, env: dict[str, str]) -> str:
    """Substitute ${VAR} placeholders. Missing vars become empty string."""
    def _sub(match):
        return env.get(match.group(1), "")
    return _TEMPLATE_VAR.sub(_sub, template)


def _arena_source_dir(default: str = "") -> str:
    """Resolve the arena runtime source-tree directory env var.

    ``MILES_ARENA_DIR`` is the preferred name for the miles port;
    ``AGISLIME_DIR`` is kept as a compatibility alias so existing deployment
    manifests keep working. A set-but-empty value falls through to the next
    candidate (same treatment as the other template-rendered env vars).
    """
    return os.environ.get("MILES_ARENA_DIR") or os.environ.get("AGISLIME_DIR") or default


# ---------------------------------------------------------------------------
# Checkpoint discovery (unchanged from previous version, kept self-contained)
# ---------------------------------------------------------------------------

def _s3_list_keys(bucket: str, prefix: str) -> list[str]:
    """Return all object keys under ``s3://bucket/prefix`` (prefix trimmed)."""
    import boto3

    s3 = boto3.client("s3")
    list_prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _looks_like_hf_checkpoint(path: str) -> bool:
    if path.startswith("s3://"):
        from miles_plugins.arena.s3_artifact import parse_s3_uri

        bucket, prefix = parse_s3_uri(path)
        keys = _s3_list_keys(bucket, prefix)
        base = prefix.strip("/")
        names = {k[len(base):].lstrip("/") for k in keys}
        return (
            "config.json" in names
            or "model.safetensors.index.json" in names
            or any(n.endswith(".safetensors") for n in names)
        )
    p = Path(path)
    return (
        (p / "config.json").exists()
        or any(p.glob("*.safetensors"))
        or (p / "model.safetensors.index.json").exists()
    )


def _s3_child_dirs(parent_uri: str) -> list[str]:
    """List immediate 'subdirectory' URIs under an ``s3://`` prefix."""
    from miles_plugins.arena.s3_artifact import parse_s3_uri

    bucket, prefix = parse_s3_uri(parent_uri)
    base = prefix.strip("/")
    list_prefix = f"{base}/" if base else ""
    children: set[str] = set()
    for key in _s3_list_keys(bucket, prefix):
        rest = key[len(list_prefix):]
        if "/" in rest:
            children.add(rest.split("/", 1)[0])
    return [f"s3://{bucket}/{list_prefix}{c}" for c in sorted(children)]


def _find_latest_hf_checkpoint(args, rollout_id: int) -> str | None:
    save_hf = getattr(args, "save_hf", None)
    if not save_hf:
        return None

    exact_path = save_hf.format(rollout_id=rollout_id, experiment_name=getattr(args, "experiment_name", ""))
    is_s3 = save_hf.startswith("s3://")

    if is_s3:
        if _looks_like_hf_checkpoint(exact_path):
            return exact_path
        # Parent is the ``.../hf`` prefix holding rollout_* subdirs.
        parent = exact_path.rsplit("/", 1)[0]
        children = [(c, c.rstrip("/").rsplit("/", 1)[-1]) for c in _s3_child_dirs(parent)]
    else:
        if Path(exact_path).is_dir() and _looks_like_hf_checkpoint(exact_path):
            return exact_path
        parent = Path(save_hf.format(rollout_id=0, experiment_name=getattr(args, "experiment_name", ""))).parent
        if not parent.is_dir():
            return None
        children = [(str(c), c.name) for c in parent.iterdir() if c.is_dir()]

    best_id = -1
    best_path = None
    for child_path, name in children:
        if not _looks_like_hf_checkpoint(str(child_path)):
            continue
        for prefix in ("rollout_", "hf_", "step_"):
            if name.startswith(prefix):
                try:
                    rid = int(name[len(prefix):])
                    if rid > best_id and rid <= rollout_id:
                        best_id = rid
                        best_path = str(child_path)
                except ValueError:
                    pass
    return best_path


def _find_latest_dcp_checkpoint(args, rollout_id: int) -> str | None:
    save_dir = getattr(args, "save", None)
    if not save_dir or not Path(save_dir).is_dir():
        return None
    best_id = -1
    best_path = None
    for child in Path(save_dir).iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            rid = int(child.name.split("_")[1])
            if rid > best_id and rid <= rollout_id:
                best_id = rid
                best_path = str(child)
        except (ValueError, IndexError):
            pass
    return best_path


def _find_convert_script() -> str | None:
    """Locate the DCP->HF conversion script.

    Miles ships the converter as ``tools/convert_torch_dist_to_hf.py`` (same
    ``--input-dir/--output-dir/--origin-hf-dir/--force`` CLI as the
    AGISlime-era ``convert_torch_dist_to_hf_parallel.py``); the parallel
    variant is kept as a compatibility fallback for runtime trees that still
    ship it.
    """
    arena_dir = _arena_source_dir()
    candidates = [
        Path(arena_dir) / "tools" / "convert_torch_dist_to_hf.py",
        # Repo-root tools/ when running from a miles source tree
        # (<repo>/miles_plugins/arena/nats_arena/eval_rollout.py -> <repo>).
        Path(__file__).resolve().parents[3] / "tools" / "convert_torch_dist_to_hf.py",
        Path(arena_dir) / "tools" / "convert_torch_dist_to_hf_parallel.py",
        Path(__file__).resolve().parents[2] / "tools" / "convert_torch_dist_to_hf_parallel.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _convert_dcp_to_hf(args, rollout_id: int) -> str | None:
    dcp_path = _find_latest_dcp_checkpoint(args, rollout_id)
    if dcp_path is None:
        return None

    save_hf = getattr(args, "save_hf", None)
    if save_hf:
        dcp_rollout_id = int(Path(dcp_path).name.split("_")[1])
        hf_output = save_hf.format(
            rollout_id=dcp_rollout_id,
            experiment_name=getattr(args, "experiment_name", ""),
        )
    else:
        hf_output = str(Path(dcp_path).parent / "hf" / Path(dcp_path).name)

    # If the HF dir is already present at the destination, reuse it.
    if _looks_like_hf_checkpoint(hf_output):
        return hf_output

    # The conversion subprocess writes safetensors to a POSIX --output-dir,
    # so when the destination is s3:// convert into a local temp dir and
    # upload afterwards (returning the s3:// URI for downstream consumers).
    # The temp dir is removed in the finally below regardless of outcome.
    s3_output = hf_output if hf_output.startswith("s3://") else None
    _convert_tmp = None
    if s3_output is not None:
        import shutil
        import tempfile

        _convert_tmp = tempfile.mkdtemp(prefix="dcp2hf_")
        hf_output = _convert_tmp

    try:
        convert_script = _find_convert_script()
        if convert_script is None:
            logger.error("DCP→HF conversion script not found.")
            return None

        hf_checkpoint = getattr(args, "hf_checkpoint", None)
        if not hf_checkpoint:
            logger.error("--hf-checkpoint not set; cannot convert DCP→HF.")
            return None

        cmd = [
            "python3", convert_script,
            "--input-dir", dcp_path,
            "--output-dir", hf_output,
            "--origin-hf-dir", hf_checkpoint,
            "--force",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logger.error("DCP→HF conversion failed: %s", result.stderr[-2000:])
                return None
            if s3_output is not None:
                # Upload the freshly-converted HF dir to S3 and hand back the URI.
                from miles_plugins.arena.s3_artifact import upload_dir

                upload_dir(hf_output, s3_output)
                logger.info("Uploaded converted HF checkpoint to %s", s3_output)
                return s3_output
            return hf_output
        except Exception as e:
            logger.error("DCP→HF conversion error: %s", e)
            return None
    finally:
        # Drop the local staging dir so per-step conversions don't accumulate
        # multi-GB HF copies on the trainer pod's disk.
        if _convert_tmp is not None:
            shutil.rmtree(_convert_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Eval coordinator Job render + apply
# ---------------------------------------------------------------------------

def _build_coordinator_env(args, rollout_id: int, hf_path: str) -> dict[str, str]:
    """Build the env-var dict used to render the eval-coordinator-job.yaml."""
    job_name = os.environ.get("JOB_NAME", "eval-job")
    user_alias = os.environ.get("USER_ALIAS", getattr(args, "user", "unknown"))
    namespace = os.environ.get("NAMESPACE", f"user-{user_alias}")
    cluster = os.environ.get("CLUSTER", "bom")

    if cluster == "bom":
        code_mount_path = "/scratch"
        code_pvc = "arena-s3-scratch"
        arenabase_path = f"/scratch/{user_alias}/code/AREnABase"
        arena_trainer_path = _arena_source_dir(f"/scratch/{user_alias}/code/AGISlime")
        default_cpu_node_type = "r5ad.24xlarge"
    else:
        code_mount_path = "/code"
        code_pvc = "fsx-claim-general--atl-2a1"
        arenabase_path = f"/code/{user_alias}-sandbox/AREnABase"
        arena_trainer_path = _arena_source_dir(f"/code/{user_alias}-sandbox/AGISlime")
        default_cpu_node_type = "r5.24xlarge"

    # All eval-pod-spec args flow from hydra config -> argparse -> args
    # (registered on the trainer parser). Per-arg fallback layering:
    #   1. args.<name>             — set by --eval-* CLI flag from hydra
    #   2. args.<sglang_fallback>  — only where it makes sense to inherit
    #                                from the rollout-side sglang config
    #   3. constant default
    # Env vars are no longer consulted: deploy.sh used to be the source of
    # truth for these but silently rendered empty strings whenever its
    # eval-vars export block was skipped (e.g. no --eval-gyms passed),
    # breaking int parses downstream. Hydra-driven config is the single
    # source of truth now.
    eval_tp_size = str(
        getattr(args, "eval_tp_size", None)
        or getattr(args, "rollout_num_gpus_per_engine", None)
        or 4
    )
    eval_gpu_node_type = (
        getattr(args, "eval_gpu_node_type", None)
        or os.environ.get("GPU_NODE_TYPE")  # set by deploy.sh, always present
        or "p6-b200.48xlarge"
    )
    gpu_nodepool = (
        getattr(args, "eval_gpu_nodepool", None)
        or os.environ.get("GPU_NODEPOOL")
        or "p6-gpu-1a"
    )

    nats_svc = os.environ.get("NATS_SVC", f"{job_name}-nats-svc")
    image = os.environ.get("IMAGE", getattr(args, "image", ""))
    dind_image = os.environ.get("DIND_IMAGE", "011550804606.dkr.ecr.us-east-1.amazonaws.com/arena-base:dind-27")
    arena_image = os.environ.get("ARENA_IMAGE", "011550804606.dkr.ecr.us-east-1.amazonaws.com/arena-base:abhanshu")

    eval_sglang_replicas = str(getattr(args, "eval_sglang_replicas", None) or 1)
    eval_context_length = str(
        getattr(args, "eval_context_length", None)
        or getattr(args, "sglang_context_length", None)
        or 65536
    )
    eval_chunked_prefill_size = str(
        getattr(args, "eval_chunked_prefill_size", None)
        or getattr(args, "sglang_chunked_prefill_size", None)
        or 8192
    )
    eval_tool_call_parser = (
        getattr(args, "eval_tool_call_parser", None)
        or getattr(args, "sglang_tool_call_parser", None)
        or "qwen3_coder"
    )
    eval_reasoning_parser = (
        getattr(args, "eval_reasoning_parser", None)
        or getattr(args, "sglang_reasoning_parser", None)
        or "qwen3"
    )
    eval_sglang_timeout = str(getattr(args, "eval_sglang_timeout", None) or 1800)
    # Per-eval-task wall-clock passed to the eval-gym ``arena run --timeout``.
    # Source of truth is the top-level ``eval_timeout`` config; the per-gym
    # ``eval-gyms.yaml`` ``timeout`` is an optional override applied later in
    # eval_coordinator._publish_and_collect. Kept < the gym consumer ack_wait
    # (timeout+300, derived in AREnABase _run_setup) so a task acks by its own
    # timeout before NATS redelivers it. Default 3000s.
    eval_timeout = str(getattr(args, "eval_timeout", None) or 3000)

    arena_max_tokens = str(getattr(args, "rollout_max_response_len", 4096))

    wandb_secret_name = os.environ.get("WANDB_SECRET_NAME", getattr(args, "wandb_secret_name", user_alias))
    # wandb_run_id is set on args by init_wandb_primary() after wandb.init().
    # The coordinator uses resume="must" with this id so its log() calls
    # attach to the trainer's wandb run rather than starting a new one.
    wandb_run_id = (
        getattr(args, "wandb_run_id", None)
        or os.environ.get("WANDB_RUN_ID", "")
        or ""
    )
    wandb_project = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT", "")
    wandb_group = getattr(args, "wandb_group", None) or os.environ.get("WANDB_GROUP", "")
    # Entity/team for OTLP project_id resolution. The coordinator and the
    # eval-gym pods it spawns build project_id as ``${WANDB_TEAM}/${WANDB_PROJECT}``;
    # without it they fall back to ``wandb.Api().default_entity`` (the user's
    # PERSONAL entity, e.g. ``rahsubb``), so eval traces orphan into
    # ``rahsubb/<project>`` instead of co-locating with the training run at
    # ``arena/<project>``. The trainer pod has WANDB_TEAM in its env (set by
    # deploy.sh); forward it here — the coordinator-job.yaml + eval-gym-workers.yaml
    # templates both reference ${WANDB_TEAM}.
    wandb_team = (
        getattr(args, "wandb_team", None)
        or os.environ.get("WANDB_TEAM", "")
        or os.environ.get("WANDB_ENTITY", "")
    )

    # Build EVAL_GYMS_CONFIG (JSON list). Source of truth is the
    # ``--eval-gyms <file>`` passed to deploy.sh, which parses the conf file
    # and exports EVAL_GYMS_CONFIG into the trainer pod env. Falls back to
    # synthesising specs from eval_datasets when the env var is absent
    # (local runs without deploy.sh).
    eval_gyms_raw = os.environ.get("EVAL_GYMS_CONFIG", "")
    if not eval_gyms_raw:
        gym_names = set()
        for ds in getattr(args, "eval_datasets", []) or []:
            if getattr(ds, "gym_name", None):
                gym_names.add(ds.gym_name)
        eval_gyms_raw = json.dumps([
            {"gym_name": n, "replicas": 4, "concurrency": 1, "max_iterations": 1000}
            for n in gym_names
        ])

    # Build EVAL_DATASETS_JSON — coordinator needs the dataset configs
    eval_datasets_list = []
    for ds in getattr(args, "eval_datasets", []) or []:
        eval_datasets_list.append({
            "name": getattr(ds, "name", ""),
            "path": getattr(ds, "path", ""),
            "gym_name": getattr(ds, "gym_name", "") or "",
            "n_samples_per_eval_prompt": getattr(ds, "n_samples_per_eval_prompt", None) or 1,
            "rm_type": getattr(ds, "rm_type", "") or "",
        })
    eval_datasets_json = json.dumps(eval_datasets_list)

    # Shared queue dir for eval metrics (mirrors trainer-side derivation in
    # miles_plugins.arena.eval_metrics_drain.get_eval_metrics_dir). This is a
    # LOCAL POSIX IPC dir (mtime heartbeats + flock), so it anchors on
    # args.save (DCP dir, stays on /scratch) — never on an s3:// save_hf.
    save_dir = getattr(args, "save", None)
    save_hf = getattr(args, "save_hf", None)
    eval_metrics_dir = ""
    if save_dir:
        eval_metrics_dir = str(Path(save_dir) / "eval_metrics")
    elif save_hf and not save_hf.startswith("s3://"):
        try:
            rollout_0 = save_hf.format(
                rollout_id=0, experiment_name=getattr(args, "experiment_name", "")
            )
            eval_metrics_dir = str(Path(rollout_0).parent.parent / "eval_metrics")
        except (KeyError, IndexError):
            pass

    return {
        "JOB_NAME": job_name,
        "STEP": str(rollout_id),
        "NAMESPACE": namespace,
        "USER_ALIAS": user_alias,
        "IMAGE": image,
        "EVAL_HF_PATH": hf_path,
        "EVAL_METRICS_DIR": eval_metrics_dir,
        "NATS_SVC": nats_svc,
        "ARENA_IMAGE": arena_image,
        "DIND_IMAGE": dind_image,
        "ARENABASE_PATH": arenabase_path,
        "AGISLIME_PATH": arena_trainer_path,
        "CODE_MOUNT_PATH": code_mount_path,
        "CODE_PVC": code_pvc,
        "GPU_NODEPOOL": gpu_nodepool,
        "EVAL_GPU_NODE_TYPE": eval_gpu_node_type,
        "CPU_NODE_TYPE": default_cpu_node_type,
        "EVAL_TP_SIZE": eval_tp_size,
        "EVAL_SGLANG_REPLICAS": eval_sglang_replicas,
        "EVAL_CONTEXT_LENGTH": eval_context_length,
        "EVAL_CHUNKED_PREFILL_SIZE": eval_chunked_prefill_size,
        "EVAL_TOOL_CALL_PARSER": eval_tool_call_parser,
        "EVAL_REASONING_PARSER": eval_reasoning_parser,
        "EVAL_SGLANG_TIMEOUT": eval_sglang_timeout,
        # Per-task eval-gym timeout (source of truth = ``eval_timeout`` config).
        "EVAL_TIMEOUT": eval_timeout,
        "ARENA_MAX_TOKENS": arena_max_tokens,
        "WANDB_SECRET_NAME": wandb_secret_name,
        "WANDB_RUN_ID": wandb_run_id,
        "WANDB_PROJECT": wandb_project,
        "WANDB_GROUP": wandb_group,
        "WANDB_TEAM": wandb_team,
        "EVAL_GYMS_CONFIG": eval_gyms_raw,
        "EVAL_DATASETS_JSON": eval_datasets_json,
        # Trajectory tracing env, inherited from the trainer pod's own env
        # (set by deploy.sh from enable_trajectory_tracing /
        # mirror_traces_to_scratch). Both the eval-coordinator-job.yaml and
        # eval-gym-workers.yaml templates reference these via ${...}; without
        # forwarding them here they render empty, so (a) the tracing master
        # switch never reaches the eval path and (b) the on-disk trajectory
        # mirror that ``mirror_traces_to_scratch`` enables for training never
        # turns on for eval — eval span trees would not be written to the
        # shared scratch traces dir alongside the training ones.
        "ARENA_DISABLE_TRACING": os.environ.get("ARENA_DISABLE_TRACING", ""),
        "ARENA_LOCAL_TRACES_DIR": os.environ.get("ARENA_LOCAL_TRACES_DIR", ""),
        # S3 artifact region — forwarded so the coordinator's S3 clients (HF
        # checkpoint copy, trace shard reads) target the artifact bucket's
        # region rather than the image-baked AWS_REGION (us-east-1).
        "S3_ARTIFACT_REGION": os.environ.get("S3_ARTIFACT_REGION", ""),
        # HaLLMark integration (S3 upload + API call)
        "HALLMARK_ENABLED": os.environ.get("HALLMARK_ENABLED", "0"),
        "HALLMARK_BENCHMARKS": os.environ.get("HALLMARK_BENCHMARKS", ""),
        "HALLMARK_S3_BUCKET": os.environ.get("HALLMARK_S3_BUCKET", ""),
        "HALLMARK_MODEL_PROFILE": os.environ.get("HALLMARK_MODEL_PROFILE", ""),
    }


def _find_template_path() -> Path:
    """Locate eval-coordinator-job.yaml in a running trainer pod."""
    candidates = [
        Path(_arena_source_dir()) / "experiments/k8s/templates/eval-coordinator-job.yaml",
        Path(__file__).resolve().parents[2] / "experiments/k8s/templates/eval-coordinator-job.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise RuntimeError(f"eval-coordinator-job.yaml not found. Tried: {candidates}")


def _apply_yaml(rendered: str, namespace: str) -> None:
    """Apply YAML via the Python kubernetes client (in-cluster service account).

    The trainer image typically does not ship with kubectl, so we use the
    kubernetes Python SDK directly. Each YAML document is applied as a
    create-or-replace per resource kind.
    """
    import yaml

    try:
        import kubernetes
    except ImportError as e:
        raise RuntimeError(
            "kubernetes Python SDK required for eval_rollout. "
            "Add `pip install kubernetes` to the trainer entrypoint."
        ) from e

    # Re-load incluster config on every call. Projected SA tokens rotate
    # (~1h on most EKS clusters); load_incluster_config() reads the token
    # once and caches it on the default API client, so a long-running
    # trainer ends up sending a stale token and hitting 401 Unauthorized
    # on subsequent eval triggers. Reading fresh from disk on every call
    # avoids that.
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException as exc:
        logger.warning("load_incluster_config failed (%s); falling back to kube_config", exc)
        try:
            kubernetes.config.load_kube_config()
        except Exception:
            raise RuntimeError(
                "Could not load Kubernetes config from in-cluster service account "
                "or local kube config. Check that the trainer pod has "
                "automountServiceAccountToken enabled."
            ) from exc

    batch = kubernetes.client.BatchV1Api()
    api_exc = kubernetes.client.exceptions.ApiException

    for doc in yaml.safe_load_all(rendered):
        if not doc:
            continue
        kind = doc.get("kind")
        meta = doc.setdefault("metadata", {})
        meta.setdefault("namespace", namespace)
        name = meta["name"]

        if kind == "Job":
            try:
                batch.create_namespaced_job(namespace, doc)
                logger.info("Created Job %s", name)
            except api_exc as exc:
                if exc.status == 409:
                    # Already exists from a prior trigger at this step (idempotent
                    # retrigger). Delete and recreate so the new Job runs.
                    logger.info("Job %s exists; deleting and recreating", name)
                    batch.delete_namespaced_job(
                        name, namespace,
                        propagation_policy="Foreground",
                    )
                    # Wait briefly for the old Job + its pod to be cleaned up
                    for _ in range(60):
                        try:
                            batch.read_namespaced_job(name, namespace)
                            time.sleep(2)
                        except api_exc as e:
                            if e.status == 404:
                                break
                    batch.create_namespaced_job(namespace, doc)
                    logger.info("Recreated Job %s", name)
                else:
                    raise
        else:
            logger.warning("Skipping unsupported kind in eval-coordinator-job: %s", kind)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """Fire-and-forget eval trigger.

    Renders eval-coordinator-job.yaml for ``rollout_id`` and applies it.
    Returns immediately with empty data — the coordinator Job runs
    independently and logs metrics directly to wandb.

    Concurrency: no cap. Multiple coordinators can run for different steps.
    Resource constraints (GPU availability) keep them queued naturally —
    pending coordinator pods sit in K8s scheduler until resources are free.

    The coordinator handles both NATS gym eval and HaLLMark benchmarks
    (if configured via --hallmark-benchmarks) in parallel.
    """
    if not evaluation:
        raise ValueError(
            "eval_rollout.generate_rollout should only be called with evaluation=True"
        )

    eval_datasets = getattr(args, "eval_datasets", None) or []
    hallmark_benchmarks = getattr(args, "hallmark_benchmarks", None)

    if not eval_datasets and not hallmark_benchmarks:
        logger.warning("No eval datasets or HaLLMark benchmarks configured; skipping eval trigger at step=%d", rollout_id)
        return RolloutFnEvalOutput(data={})

    # Resolve HF checkpoint for this step from --save-hf (set by entrypoint.sh
    # to $CKPT_DIR/hf/rollout_{rollout_id}). Fall back to DCP→HF conversion
    # if the HF directory hasn't been written yet.
    hf_path = _find_latest_hf_checkpoint(args, rollout_id)
    if hf_path is None:
        hf_path = _convert_dcp_to_hf(args, rollout_id)

    if hf_path is None:
        logger.warning(
            "No HF checkpoint resolvable for eval at rollout_id=%d; "
            "skipping eval trigger (training continues).", rollout_id,
        )
        return RolloutFnEvalOutput(data={})

    logger.info(
        "Triggering fire-and-forget eval coordinator Job for step=%d (checkpoint=%s)",
        rollout_id, hf_path,
    )

    # Render and apply the coordinator Job. Template resolution stays inside the
    # guard: a tree without experiments/k8s/templates (e.g. MILES_ARENA_DIR left
    # at the launcher-baked miles checkout) raises RuntimeError, and the eval
    # trigger must log-and-skip rather than propagate into the trainer.
    try:
        template_path = _find_template_path()
        template = template_path.read_text()
        env = _build_coordinator_env(args, rollout_id, hf_path)
        rendered = _render(template, env)
        _apply_yaml(rendered, namespace=env["NAMESPACE"])
    except Exception as e:
        logger.error(
            "Failed to render/apply eval coordinator Job for step=%d: %s", rollout_id, e, exc_info=True
        )
        # Don't crash the trainer — just log and skip
        return RolloutFnEvalOutput(data={})

    logger.info(
        "Eval coordinator Job submitted: %s-eval-coord-step-%d. "
        "Trainer continues; metrics will appear in wandb when eval completes.",
        env["JOB_NAME"], rollout_id,
    )

    # Fire-and-forget: return empty data immediately
    return RolloutFnEvalOutput(data={})
