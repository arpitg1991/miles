"""Standalone eval coordinator — runs as a K8s Job pod, fire-and-forget from trainer.

Workflow per step:
  1. Resolve own Job (for OwnerReferences) so spawned Deployments cascade-delete
     when this Job completes.
  2. Ensure NATS eval streams exist (idempotent — survives trainer restarts).
  3. Render and apply eval-sglang Deployment + Service for this step (with OwnerRef).
  4. Render and apply eval-gym-worker Deployments for this step (with OwnerRef).
  5. Wait for eval SGLang readiness.
  6. Publish tasks to ``arena.eval.tasks.{step}.{gym}``.
  7. Collect results from ``arena.eval.results.{step}`` until all received
     (no timeout — pods stay alive until done; trainer can move on independently).
  8. Compute per-dataset metrics, log to the trainer's wandb run via resume="must".
  9. Exit. K8s GC cascade-deletes the eval-sglang + eval-gym Deployments.

Environment variables:
  Required: JOB_NAME, NAMESPACE, EVAL_STEP, EVAL_HF_PATH, NATS_URL,
            EVAL_GYMS_CONFIG (JSON), EVAL_DATASETS_JSON (JSON),
            IMAGE, ARENA_IMAGE, DIND_IMAGE, ARENABASE_PATH,
            AGISLIME_PATH (or its preferred alias MILES_ARENA_PATH),
            CODE_MOUNT_PATH, CODE_PVC, GPU_NODEPOOL, EVAL_GPU_NODE_TYPE,
            CPU_NODE_TYPE, EVAL_TP_SIZE, EVAL_SGLANG_REPLICAS,
            EVAL_CONTEXT_LENGTH, EVAL_CHUNKED_PREFILL_SIZE,
            EVAL_TOOL_CALL_PARSER, EVAL_REASONING_PARSER, ARENA_MAX_TOKENS,
            USER_ALIAS, WANDB_SECRET_NAME
  Optional: WANDB_RUN_ID, WANDB_PROJECT, WANDB_GROUP (for wandb logging).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_coordinator")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVAL_TASKS_STREAM = "ARENA_EVAL_TASKS"
_EVAL_RESULTS_STREAM = "ARENA_EVAL_RESULTS"
_EVAL_TASKS_SUBJECT_PREFIX = "arena.eval.tasks"  # then .<step>.<gym>
_EVAL_RESULTS_SUBJECT_PREFIX = "arena.eval.results"  # then .<step>

_SGLANG_READY_POLL_INTERVAL = 10
# Default if EVAL_SGLANG_TIMEOUT env var is not set. Read at usage time.
_SGLANG_READY_TIMEOUT_DEFAULT = 1800  # 30 min

# Result-collection poll interval. There is NO overall eval timeout — the
# coordinator stays alive until all expected results arrive. Pods can be
# manually deleted to abort.
_RESULT_POLL_INTERVAL = 5
_PROGRESS_LOG_INTERVAL = 60  # log progress every minute

# How long to wait for the eval coordinator's own Job UID to appear via the
# K8s API (sometimes there's a brief lag after pod scheduling).
_JOB_UID_DISCOVERY_TIMEOUT = 60


# ---------------------------------------------------------------------------
# OTLP tracer provider (eval coordinator publisher side)
# ---------------------------------------------------------------------------


def _make_wandb_run_id_span_processor(wandb_run_id: str):
    """Return a SpanProcessor that injects ``wandb.wb_run_id`` on every span.

    Inlined here for parity with nats_rollout.py — the factory keeps the
    eval coordinator self-contained without depending on AREnABase being
    importable at the call site.

    W&B's weave OTel ingester reads this specific span-attribute key and
    populates the call's ``wb_run_id`` column — what the trace UI's
    "filter by run" dropdown queries. The server auto-formats a bare
    run-id (no ``/`` or ``:``) to ``{entity}/{project}:{run_id}``.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    class _WandbRunIdSpanProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None):  # noqa: ARG002
            try:
                span.set_attribute("wandb.wb_run_id", wandb_run_id)
            except Exception:
                pass

    return _WandbRunIdSpanProcessor()


def _make_urandom_id_generator():
    """Return an OTel IdGenerator backed by ``os.urandom``.

    Mirrors nats_rollout._make_urandom_id_generator. Decouples span/trace
    ids from Python's ``random`` module — the trainer re-seeds it
    deterministically in megatron init, which causes the default ``RandomIdGenerator`` to
    emit identical span_ids across runs that share args.seed, collapsing
    multiple runs' rollouts into one trace in W&B's UI.
    """
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    class _UrandomIdGenerator(IdGenerator):
        def generate_span_id(self) -> int:
            sid = int.from_bytes(os.urandom(8), "big")
            return sid or 1  # 0 is INVALID_SPAN_ID

        def generate_trace_id(self) -> int:
            tid = int.from_bytes(os.urandom(16), "big")
            return tid or 1  # 0 is INVALID_TRACE_ID

    return _UrandomIdGenerator()


def _make_local_dir_span_exporter(root: str):
    """Return a SpanExporter that appends this process's spans to
    ``<root>/<host>-<pid>.jsonl`` (one span JSON per line).

    This is an inline, stdlib-only port of AREnABase's
    ``amzn_arena_base.utils.tracing.local_exporter.LocalDirSpanExporter``.
    We inline it (rather than import it) for the SAME reason the wb_run_id
    span processor and urandom id generator are inlined above: the coord
    pod runs the trainer image, which does NOT ship ``aiodocker``. Importing
    anything under ``amzn_arena_base`` runs the top-level package
    ``__init__`` (``from amzn_arena_base.environments import Vulcan`` →
    ``utils/docker/container`` → ``import aiodocker``), so
    ``from amzn_arena_base.utils.tracing.local_exporter import ...`` raises
    ModuleNotFoundError and the coord's ``eval.step.<N>`` root span never
    reaches the on-disk mirror.

    ONE APPEND-ONLY FILE PER PROCESS: each writer (the coord, and every
    eval-gym pod / forked worker) appends to its OWN ``<host>-<pid>.jsonl``.
    The filename has NO ``trace_id`` — every span line carries its own — so
    a process writes all its spans into one file that only IT touches.
    Single-writer + append means (a) no NFS shared-file clobber (the old
    shared ``<trace_id>.json`` + ``fcntl.flock`` merge lost spans across
    nodes) and (b) O(1) per span with a bounded file count even over long
    training runs. Reassemble a trace offline via
    ``amzn_arena_base...local_exporter.merge_trace_shards`` /
    ``iter_traces`` (group lines by ``trace_id``).

    This MUST stay in lock-step with the AREnABase exporter's on-disk format
    (append-only ``<host>-<pid>.jsonl``, one span JSON per line): the coord
    and the gym pods write shards into the SAME shared scratch dir, and both
    are reassembled by the same ``merge_trace_shards`` reader.
    """
    import re as _re
    import socket as _socket
    import threading
    from pathlib import Path

    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    def _shard_filename():
        host = _re.sub(r"[^A-Za-z0-9_.-]", "_", _socket.gethostname() or "host")
        return f"{host}-{os.getpid()}.jsonl"

    def _is_s3_uri(r) -> bool:
        return isinstance(r, str) and r.startswith("s3://")

    def _parse_s3_uri(uri: str):
        rest = uri[len("s3://"):]
        bucket, _, key_prefix = rest.partition("/")
        return bucket, key_prefix.rstrip("/")

    class _LocalDirSpanExporter(SpanExporter):
        def __init__(self, root_dir: str) -> None:
            # When root_dir is an ``s3://bucket/prefix`` URI, write batches
            # to S3 via the AWS API directly (bypassing the S3-Files NFS
            # mount). S3 has no append, so each export() batch becomes its
            # OWN immutable object ``<prefix>/<host>-<pid>-<seq>.jsonl`` —
            # readers still glob ``*.jsonl`` and group by trace_id. MUST stay
            # in lock-step with AREnABase's LocalDirSpanExporter S3 path.
            self._is_s3 = _is_s3_uri(root_dir)
            self._root = root_dir if self._is_s3 else Path(root_dir)
            self._lock = threading.Lock()
            self._shutdown = False
            self._ensured = False
            self._seq = 0
            self._s3_client = None
            self._s3_bucket = ""
            self._s3_prefix = ""
            if self._is_s3:
                self._s3_bucket, self._s3_prefix = _parse_s3_uri(root_dir)
            # Per-process filename so each writer owns its own append-only
            # file; recomputed after fork (new pid) so a child never appends
            # to the parent's file.
            self._filename = _shard_filename()
            if hasattr(os, "register_at_fork"):
                os.register_at_fork(after_in_child=self._reinit_lock)

        def _reinit_lock(self) -> None:
            self._lock = threading.Lock()
            self._filename = _shard_filename()
            # boto3 clients are not fork-safe; drop the inherited one and
            # reset the per-process seq so child object keys don't collide.
            self._seq = 0
            self._s3_client = None
            self._ensured = False

        def _get_s3_client(self):
            if self._s3_client is None:
                import boto3
                self._s3_client = boto3.client("s3")
            return self._s3_client

        def _ensure_root(self) -> None:
            if not self._ensured:
                self._root.mkdir(parents=True, exist_ok=True)
                self._ensured = True

        def _export_s3(self, spans) -> SpanExportResult:
            base = self._filename[: -len(".jsonl")]
            seq = self._seq
            self._seq += 1
            key = f"{base}-{seq:06d}.jsonl"
            if self._s3_prefix:
                key = f"{self._s3_prefix}/{key}"
            body = "".join(
                span.to_json(indent=None).replace("\n", " ") + "\n" for span in spans
            ).encode("utf-8")
            self._get_s3_client().put_object(
                Bucket=self._s3_bucket, Key=key, Body=body
            )
            return SpanExportResult.SUCCESS

        def export(self, spans) -> SpanExportResult:
            if self._shutdown:
                return SpanExportResult.FAILURE
            try:
                with self._lock:
                    if self._is_s3:
                        return self._export_s3(spans)
                    self._ensure_root()
                    target = self._root / self._filename
                    # Append one single-line JSON per span. Each line stands
                    # alone (no on-disk nesting — the tree is rebuilt from
                    # parent_id on read), so a torn final line from a crash
                    # costs at most one span, never the whole file.
                    with target.open("a", encoding="utf-8") as f:
                        for span in spans:
                            f.write(span.to_json(indent=None).replace("\n", " ") + "\n")
            except OSError as exc:
                logger.error("Inline LocalDirSpanExporter write failed: %s", exc)
                return SpanExportResult.FAILURE
            except Exception as exc:  # boto3/network errors on the S3 path
                logger.error("Inline LocalDirSpanExporter S3 write failed: %s", exc)
                return SpanExportResult.FAILURE
            return SpanExportResult.SUCCESS

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

        def shutdown(self) -> None:
            self._shutdown = True

    return _LocalDirSpanExporter(root)


def _install_otlp_tracer_provider() -> None:
    """Install an OTLP TracerProvider in this coordinator pod so the
    ``eval.step.<N>`` and per-row ``task.<id>`` parent spans actually
    export to W&B.

    Uses the same per-experiment project as the training trainer + gym
    workers (``arena/${EXPERIMENT_NAME}``) so eval traces sit alongside
    training traces in one project, but their distinct root span
    (``eval.step.<N>``) keeps each eval run as its own trace tree —
    filterable and visually separate from the training rollouts.

    No-op if WANDB_API_KEY is unset, the SDK is missing, or a non-proxy
    provider is already installed.
    """
    import base64

    from opentelemetry import trace as _trace

    provider = _trace.get_tracer_provider()
    if type(provider).__name__ != "ProxyTracerProvider":
        return

    if os.environ.get("ARENA_DISABLE_TRACING") == "1":
        logger.info("ARENA_DISABLE_TRACING set; eval-coordinator OTLP tracing disabled")
        return

    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        logger.info("WANDB_API_KEY unset; eval-coordinator OTLP tracing disabled")
        return

    base_url = os.environ.get("WANDB_BASE_URL", "https://mega.wandb.agi.amazon.dev")
    # W&B's OTLP ingester requires ``project_id`` in ``entity/project``
    # form — plain names 422. Mirror the trainer's wandb.init(entity=...) so
    # eval traces co-locate with the training run. See the matching
    # paragraph in nats_rollout._install_otlp_tracer_provider.
    project_id = os.environ.get("WANDB_PROJECT_ID")
    if not project_id:
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_team = os.environ.get("WANDB_TEAM") or os.environ.get("WANDB_ENTITY")
        if wandb_project:
            if not wandb_team:
                try:
                    import wandb
                    wandb_team = wandb.Api().default_entity or ""
                except Exception as e:
                    logger.warning(
                        "Could not resolve W&B default entity (%s); "
                        "falling back to ``arena/<experiment>`` for OTel project_id",
                        e,
                    )
                    wandb_team = ""
            if wandb_team:
                project_id = f"{wandb_team}/{wandb_project}"
        if not project_id:
            project_id = (
                f"arena/{os.environ['EXPERIMENT_NAME']}"
                if os.environ.get("EXPERIMENT_NAME")
                else f"arena/{os.environ['JOB_NAME']}"
                if os.environ.get("JOB_NAME")
                else "arena/trajectories"
            )
    auth = base64.b64encode(f"api:{api_key}".encode()).decode()

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning(
            "opentelemetry SDK missing in eval coordinator (%s); "
            "``eval.step.<N>`` / ``task.<id>`` parent spans will not emit. "
            "Gym worker rollouts will still be traced but as detached roots.",
            exc,
        )
        return

    exporter = OTLPSpanExporter(
        endpoint=f"{base_url}/traces/otel/v1/traces",
        headers={
            "Authorization": f"Basic {auth}",
            "project_id": project_id,
        },
    )
    # ``id_generator`` decouples span/trace ids from Python's ``random``
    # which the trainer re-seeds deterministically in megatron init — see
    # ``_make_urandom_id_generator`` above for the full explanation.
    new_provider = TracerProvider(id_generator=_make_urandom_id_generator())
    # Stamp ``wandb.wb_run_id`` as a SPAN attribute on every span so W&B's
    # weave ingester populates the call's ``wb_run_id`` column. Factory
    # is inlined below for parity with nats_rollout.py — keeps eval
    # coordinator self-contained, no cross-package import.
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    if wandb_run_id:
        new_provider.add_span_processor(
            _make_wandb_run_id_span_processor(wandb_run_id)
        )
    new_provider.add_span_processor(BatchSpanProcessor(exporter))

    # Optional local-disk mirror — same env-var contract AREnABase
    # ``setup_tracing`` uses. When ``ARENA_LOCAL_TRACES_DIR`` is set,
    # attach a second BSP that appends this process's spans to
    # ``<host>-<pid>.jsonl`` under the directory. The coord pod emits root
    # spans (``eval.step.<N>`` / ``task.<id>``) which are the heavy parents
    # of every eval rollout — mirroring them is what lets the on-disk trees
    # re-root under the eval step instead of orphaning the rollout subtrees.
    local_traces_dir = os.environ.get("ARENA_LOCAL_TRACES_DIR")
    if local_traces_dir:
        try:
            # Inline exporter — must NOT import from amzn_arena_base, whose
            # package __init__ pulls in aiodocker (absent in the trainer
            # image this coord pod runs). See _make_local_dir_span_exporter.
            new_provider.add_span_processor(
                BatchSpanProcessor(_make_local_dir_span_exporter(local_traces_dir))
            )
        except Exception as exc:  # noqa: BLE001 — never let the mirror break OTLP
            logger.warning(
                "ARENA_LOCAL_TRACES_DIR=%s but local mirror setup failed "
                "(%s); local mirror disabled, OTLP still active",
                local_traces_dir, exc,
            )
            local_traces_dir = None

    _trace.set_tracer_provider(new_provider)
    local_msg = (
        f"; local traces -> {local_traces_dir}/<host>-<pid>.jsonl"
        if local_traces_dir else ""
    )
    logger.info(
        "Eval coordinator OTLP tracing enabled (project=%s, run_id=%s, endpoint=%s/traces/otel/v1/traces)%s",
        project_id, os.environ.get("WANDB_RUN_ID", "<unset>"), base_url, local_msg,
    )


# ---------------------------------------------------------------------------
# Self-discovery: find own Job for OwnerReferences
# ---------------------------------------------------------------------------

def _load_k8s():
    import kubernetes
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException:
        kubernetes.config.load_kube_config()
    return kubernetes


def _get_own_job_owner_ref(k8s, namespace: str, job_name: str) -> dict[str, Any]:
    """Return the OwnerReference dict to attach to spawned resources.

    Spawns of Deployments will set this Job as their owner so they
    cascade-delete when the Job completes.
    """
    batch = k8s.client.BatchV1Api()
    api_exc = k8s.client.exceptions.ApiException
    start = time.time()
    while True:
        try:
            job = batch.read_namespaced_job(name=job_name, namespace=namespace)
            return {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "name": job.metadata.name,
                "uid": job.metadata.uid,
                "controller": True,
                "blockOwnerDeletion": True,
            }
        except api_exc as exc:
            if time.time() - start > _JOB_UID_DISCOVERY_TIMEOUT:
                raise RuntimeError(
                    f"Could not resolve own Job {namespace}/{job_name} "
                    f"for OwnerReferences after {_JOB_UID_DISCOVERY_TIMEOUT}s: {exc}"
                ) from exc
            time.sleep(2)


# ---------------------------------------------------------------------------
# Template rendering & apply
# ---------------------------------------------------------------------------

_TEMPLATE_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _render(template: str, env: dict[str, str]) -> str:
    """Substitute ${VAR} placeholders. Missing vars become empty string."""
    def _sub(match):
        return env.get(match.group(1), "")
    return _TEMPLATE_VAR.sub(_sub, template)


def _split_yaml_docs(rendered: str) -> list[dict[str, Any]]:
    """Split a multi-doc YAML string into a list of parsed documents."""
    import yaml
    return [d for d in yaml.safe_load_all(rendered) if d]


def _delete_eval_infra(
    k8s,
    namespace: str,
    job_name: str,
    step: str,
    eval_gyms: list[dict[str, Any]],
) -> None:
    """Explicitly delete every K8s resource the coord created for this step.

    OwnerReferences alone aren't reliable here: the coord pod's owning
    Job stays in ``Completed`` state (not Deleted) after we exit, and on
    most K8s versions GC only cascade-deletes when the OWNER itself is
    deleted. So the eval-sglang Deployment + Service and the per-gym
    eval-gym Deployments would linger, holding GPU + CPU until the
    Job's TTL expired (or forever, if no TTL was set).

    This function removes them ourselves: deterministic names from the
    templates, ``propagation_policy="Background"`` so we don't block on
    pod termination, idempotent on 404 so a second call is harmless.
    """
    apps_v1 = k8s.client.AppsV1Api()
    core_v1 = k8s.client.CoreV1Api()
    api_exc = k8s.client.exceptions.ApiException

    delete_options = k8s.client.V1DeleteOptions(
        propagation_policy="Background",
    )

    def _delete_deployment(name: str) -> None:
        try:
            apps_v1.delete_namespaced_deployment(
                name=name, namespace=namespace, body=delete_options,
            )
            logger.info("Deleted Deployment %s", name)
        except api_exc as exc:
            if exc.status == 404:
                logger.info("Deployment %s already gone (404)", name)
            else:
                logger.warning("Failed to delete Deployment %s: %s", name, exc)

    def _delete_service(name: str) -> None:
        try:
            core_v1.delete_namespaced_service(
                name=name, namespace=namespace, body=delete_options,
            )
            logger.info("Deleted Service %s", name)
        except api_exc as exc:
            if exc.status == 404:
                logger.info("Service %s already gone (404)", name)
            else:
                logger.warning("Failed to delete Service %s: %s", name, exc)

    # eval-sglang Deployment + Service (names from eval-sglang-server.yaml)
    _delete_deployment(f"{job_name}-eval-sglang-step-{step}")
    _delete_service(f"{job_name}-eval-sglang-svc-step-{step}")

    # eval-gym Deployment per gym (name from eval-gym-workers.yaml)
    for gym_cfg in eval_gyms:
        gym_name = gym_cfg.get("gym_name", "")
        if not gym_name:
            continue
        gym_slug = gym_name.replace("_", "-")
        _delete_deployment(f"{job_name}-eval-gym-{gym_slug}-step-{step}")


def _apply_with_owner(
    k8s,
    docs: list[dict[str, Any]],
    namespace: str,
    owner_ref: dict[str, Any],
) -> None:
    """Apply each doc with OwnerReference attached. Idempotent (delete+recreate)."""
    apps_v1 = k8s.client.AppsV1Api()
    core_v1 = k8s.client.CoreV1Api()
    api_exc = k8s.client.exceptions.ApiException

    for doc in docs:
        kind = doc.get("kind")
        meta = doc.setdefault("metadata", {})
        meta.setdefault("namespace", namespace)
        # Attach owner so this resource cascade-deletes with the Job
        meta["ownerReferences"] = [owner_ref]

        name = meta["name"]

        if kind == "Deployment":
            try:
                apps_v1.create_namespaced_deployment(namespace, doc)
                logger.info("Created Deployment %s", name)
            except api_exc as exc:
                if exc.status == 409:
                    # Already exists from a prior run/retry — replace it
                    apps_v1.delete_namespaced_deployment(name, namespace)
                    time.sleep(2)
                    apps_v1.create_namespaced_deployment(namespace, doc)
                    logger.info("Recreated Deployment %s", name)
                else:
                    raise
        elif kind == "Service":
            try:
                core_v1.create_namespaced_service(namespace, doc)
                logger.info("Created Service %s", name)
            except api_exc as exc:
                if exc.status == 409:
                    core_v1.delete_namespaced_service(name, namespace)
                    time.sleep(2)
                    core_v1.create_namespaced_service(namespace, doc)
                    logger.info("Recreated Service %s", name)
                else:
                    raise
        else:
            logger.warning("Skipping unsupported kind: %s", kind)


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def _find_templates_dir() -> Path:
    """Locate the K8s template directory.

    Checks MILES_ARENA_PATH (preferred) / AGISLIME_PATH (compatibility
    alias) + /experiments/k8s/templates first (which is what the coordinator
    pod sees if the templates are on the shared code volume).
    """
    arena_path = os.environ.get("MILES_ARENA_PATH") or os.environ.get("AGISLIME_PATH", "")
    candidates = [
        Path(arena_path) / "experiments/k8s/templates" if arena_path else None,
        Path("/scratch") / os.environ.get("USER_ALIAS", "") / "code/AGISlime/experiments/k8s/templates",
        Path(__file__).resolve().parents[2] / "experiments/k8s/templates",
    ]
    for c in candidates:
        if c and c.is_dir():
            return c
    raise RuntimeError(f"Could not locate K8s templates dir. Tried: {candidates}")


# ---------------------------------------------------------------------------
# NATS stream/consumer setup
# ---------------------------------------------------------------------------

async def _ensure_nats_streams(nats_url: str, step: str):
    """Ensure the eval streams exist with wildcard subjects.

    - ARENA_EVAL_TASKS subscribes to "arena.eval.tasks.>" (work-queue).
    - ARENA_EVAL_RESULTS subscribes to "arena.eval.results.>" (limits).

    Per-step subjects/consumers route within these streams. We do NOT purge
    streams here — that would clobber other concurrent evals. Streams are
    created with max_age so orphaned tasks/results auto-clean.
    """
    import nats
    from nats.js.api import RetentionPolicy, StreamConfig

    nc = await nats.connect(nats_url, max_reconnect_attempts=-1, reconnect_time_wait=2)
    js = nc.jetstream()

    tasks_pattern = f"{_EVAL_TASKS_SUBJECT_PREFIX}.>"
    results_pattern = f"{_EVAL_RESULTS_SUBJECT_PREFIX}.>"

    try:
        info = await js.stream_info(_EVAL_TASKS_STREAM)
        # Update subjects if stream was created with old pattern
        existing = list(info.config.subjects or [])
        if tasks_pattern not in existing:
            await js.update_stream(config=StreamConfig(
                name=_EVAL_TASKS_STREAM,
                subjects=[tasks_pattern],
                retention=RetentionPolicy.WORK_QUEUE,
                max_age=14400,  # 4h in seconds (nats-py converts to ns internally)
            ))
            logger.info("Updated %s subjects to %s", _EVAL_TASKS_STREAM, tasks_pattern)
    except Exception:
        await js.add_stream(config=StreamConfig(
            name=_EVAL_TASKS_STREAM,
            subjects=[tasks_pattern],
            retention=RetentionPolicy.WORK_QUEUE,
            max_age=14400,  # 4h in seconds
        ))
        logger.info("Created stream %s (subjects=%s)", _EVAL_TASKS_STREAM, tasks_pattern)

    try:
        info = await js.stream_info(_EVAL_RESULTS_STREAM)
        existing = list(info.config.subjects or [])
        if results_pattern not in existing:
            await js.update_stream(config=StreamConfig(
                name=_EVAL_RESULTS_STREAM,
                subjects=[results_pattern],
                retention=RetentionPolicy.LIMITS,
                max_age=14400,  # 4h in seconds
                max_msg_size=134_217_728,
            ))
            logger.info("Updated %s subjects to %s", _EVAL_RESULTS_STREAM, results_pattern)
    except Exception:
        await js.add_stream(config=StreamConfig(
            name=_EVAL_RESULTS_STREAM,
            subjects=[results_pattern],
            retention=RetentionPolicy.LIMITS,
            max_age=14400,  # 4h in seconds
            max_msg_size=134_217_728,
        ))
        logger.info("Created stream %s (subjects=%s)", _EVAL_RESULTS_STREAM, results_pattern)

    await nc.close()


# ---------------------------------------------------------------------------
# SGLang readiness
# ---------------------------------------------------------------------------

def _wait_for_sglang_ready(svc_name: str, namespace: str) -> None:
    """Poll the eval SGLang service until it's ready.

    Timeout is configurable via EVAL_SGLANG_TIMEOUT env var (seconds).
    Per user spec: if SGLang fails to start within the timeout, the
    coordinator pod fails and exits; K8s GC cascade-deletes the
    eval-sglang + eval-gym Deployments. The trainer is unaffected.
    """
    # os.environ.get returns "" (not the default) when the var is set-but-empty,
    # which can happen if deploy.sh didn't export it (e.g. --eval-gyms not passed
    # so the export block at deploy.sh:619-635 was skipped) — envsubst then
    # renders the trainer template with EVAL_SGLANG_TIMEOUT="". Treat empty
    # string the same as missing.
    timeout = int(os.environ.get("EVAL_SGLANG_TIMEOUT") or _SGLANG_READY_TIMEOUT_DEFAULT)
    health_url = f"http://{svc_name}.{namespace}.svc.cluster.local:30000/health"
    logger.info("Waiting for eval SGLang at %s (timeout=%ds)...", health_url, timeout)
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("Eval SGLang ready after %.0fs", time.time() - start)
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(_SGLANG_READY_POLL_INTERVAL)
    raise TimeoutError(f"Eval SGLang not ready after {timeout}s at {health_url}")


# ---------------------------------------------------------------------------
# Dataset loading + task publishing + result collection
# ---------------------------------------------------------------------------

def _load_eval_dataset(dataset_cfg: dict) -> list[dict[str, Any]]:
    """Load eval dataset rows from a local path or lakefs:// URI."""
    from miles_plugins.arena.nats_arena.data_source import _resolve_manifest_path

    resolved = _resolve_manifest_path(dataset_cfg["path"])
    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path} (from {dataset_cfg['path']})")

    rows = []
    with open(path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at %s:%d: %s", path, line_num + 1, e)
    logger.info("Loaded %d eval tasks from %s", len(rows), path)
    return rows


def _row_to_task(
    row: dict, dataset_cfg: dict, task_index: int, n_samples: int = 1
) -> dict[str, Any]:
    """Convert a manifest row to a NATS task message (lakeFS pointer format).

    ``n_samples`` rides on the task as ``msg["n_samples"]`` so the eval gym
    pod runs the WHOLE group of rollouts in one pod (via its AsyncWorker fork
    pool), mirroring the trainer's batched model. One task per prompt instead
    of replicating the prompt into N ``-s<idx>`` tasks scattered across pods.
    """
    from miles_plugins.arena.nats_arena.message_format import build_task_message

    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    lakefs_uri = row.get("lakefs_uri") or metadata.get("lakefs_uri", "")
    lakefs_commit_id = row.get("lakefs_commit_id") or metadata.get("lakefs_commit_id", "")
    if not lakefs_uri:
        raise ValueError(
            f"Eval row {task_index} in dataset {dataset_cfg['name']!r} missing 'lakefs_uri'."
        )

    instance_id = str(
        row.get("id")
        or row.get("task_id")
        or metadata.get("instance_id")
        or _task_id_from_lakefs_uri(lakefs_uri)
        or f"eval-{dataset_cfg['name']}-{task_index}"
    )

    gym_name = dataset_cfg.get("gym_name") or metadata.get("gym_name", "")

    return build_task_message(
        instance_id=instance_id,
        lakefs_uri=lakefs_uri,
        lakefs_commit_id=lakefs_commit_id,
        n_samples=n_samples,
        metadata={
            **metadata,
            "eval": True,
            "eval_dataset": dataset_cfg["name"],
            "label": row.get("label", row.get("answer", "")),
        },
        gym_name=gym_name,
    )


def _task_id_from_lakefs_uri(lakefs_uri: str) -> str:
    trimmed = lakefs_uri.rstrip("/")
    parts = trimmed.split("/")
    try:
        idx = len(parts) - 1 - parts[::-1].index("tasks")
    except ValueError:
        return parts[-1] if parts else ""
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return parts[-1] if parts else ""


async def _publish_and_collect(
    eval_datasets: list[dict],
    nats_url: str,
    step: str,
) -> dict[str, dict[str, Any]]:
    """Publish all eval tasks, then block collecting until every task returns.

    No overall timeout. The coordinator pod stays alive as long as needed.

    OTel span hierarchy emitted by this function:

        {gym}/eval.step.<step>                     (one per gym per coord run)
        └── {gym}/task.<instance_id>               (gym-emitted; child via W3C
            └── rollout                             traceparent injected here)
                ├── rollout.setup
                ├── chat_completion / tool_call
                └── rollout.grade

    Only the ``eval.step.<N>`` root is opened coordinator-side. The
    ``task.<id>`` span and everything below it are emitted by the gym
    worker's GenerationDispatcher, which re-parents its ``{gym}/task.<id>``
    span under this step via the injected ``task["trace_context"]`` carrier.
    The ``task.<id>`` subtree is therefore identical to a training
    trajectory. The distinct ``eval.step.<N>`` root keeps eval traces
    visually separate from training rollouts in the shared W&B project.
    """
    import nats
    from nats.js.api import ConsumerConfig
    from opentelemetry import trace
    from opentelemetry.propagate import inject

    from miles_plugins.arena.nats_arena.message_format import (
        extract_trajectories,
        parse_result,
        serialize_task,
    )

    nc = await nats.connect(nats_url, max_reconnect_attempts=-1, reconnect_time_wait=2)
    js = nc.jetstream()

    results_subject = f"{_EVAL_RESULTS_SUBJECT_PREFIX}.{step}"
    consumer_name = f"slime-eval-coord-step-{step}"

    # Create per-step results consumer (filtered to this step's subject only)
    try:
        await js.delete_consumer(_EVAL_RESULTS_STREAM, consumer_name)
    except Exception:
        pass
    await js.add_consumer(
        stream=_EVAL_RESULTS_STREAM,
        config=ConsumerConfig(
            durable_name=consumer_name,
            filter_subject=results_subject,
            ack_policy="explicit",
            max_deliver=3,
            ack_wait=3600,
        ),
    )

    # Publish tasks
    total_tasks = 0
    task_registry: dict[str, str] = {}  # task_id -> dataset_name
    # Wall-clock publish time per task_id. Drives the DLQ sweep below
    # so a poison prompt that exhausts NATS's max_deliver=3 retries
    # doesn't deadlock the eval forever on `while collected < total_tasks`.
    publish_times: dict[str, float] = {}
    # Set of task_ids whose result has already been counted into
    # `collected`. Required because of two distinct duplicate sources:
    #   1. NATS redelivery within max_deliver budget (worker crashed
    #      after fetching but before acking).
    #   2. The reopen path below (`_reopen_psub`) recreates the durable
    #      consumer if it was lost — the new consumer starts at
    #      deliver_policy=all and replays everything still in
    #      ARENA_EVAL_RESULTS that matches its filter_subject.
    # Without dedup, both paths inflate `collected` past `total_tasks`
    # or double-count rewards. With dedup, late duplicates are silently
    # acked and dropped.
    seen_task_ids: set[str] = set()
    # Tasks the DLQ sweep counted as missing-result. Tracked separately so a
    # late/undrained REAL result arriving afterwards (in a drain pass) can
    # OVERRIDE the synthetic miss: _handle_result_msg treats a tid that is in
    # dlq_swept but has no real reward yet as fresh (re-credits the reward,
    # does NOT double-count collected).
    dlq_swept: set[str] = set()

    # --- Bounded in-flight publishing -------------------------------------
    # Instead of publishing ALL n_prompts x n_samples tasks to NATS upfront
    # (which left e.g. 312/352 sitting in the work-queue while only ~replicas
    # ran), we keep the number of OUTSTANDING tasks capped at roughly the gym
    # worker capacity and publish a new task only as earlier ones finish.
    # This avoids a giant backlog whose queued tasks age against the DLQ
    # deadline while starved behind a few slow ones.
    #
    # Cap = sum over eval gyms of (replicas * concurrency), read from
    # EVAL_GYMS_CONFIG (the same config that sizes the eval-gym Deployments).
    # A small headroom multiplier keeps every worker fed (one queued task
    # ready the instant a worker acks) without rebuilding the old backlog.
    # Overridable via EVAL_MAX_INFLIGHT.
    try:
        _egc = json.loads(os.environ.get("EVAL_GYMS_CONFIG", "[]") or "[]")
        _gym_capacity = sum(
            int(g.get("replicas", 4)) * int(g.get("concurrency", 1)) for g in _egc
        ) or 1
    except (ValueError, TypeError):
        _gym_capacity = 1
    # ``or "<default>"`` (not just the get() default) so an env var that is
    # PRESENT but EMPTY falls back too. The eval-coordinator-job template wires
    # ``value: "${EVAL_INFLIGHT_HEADROOM}"``, and when deploy.sh doesn't export
    # the var, envsubst renders it to "" — os.environ.get(key, "1.5") then
    # returns "" (the default only fires for a MISSING key), and float("")
    # raised ValueError, killing the eval before any task dispatched.
    _inflight_headroom = float(os.environ.get("EVAL_INFLIGHT_HEADROOM") or "1")
    max_inflight = int(os.environ.get("EVAL_MAX_INFLIGHT") or str(
        max(1, int(_gym_capacity * _inflight_headroom))
    ))
    # FIFO of tasks built-but-not-yet-published: (subject, task_dict).
    pending_tasks: deque[tuple[str, dict]] = deque()

    async def _publish_one() -> bool:
        """Publish the next pending task (if any). Returns True if published."""
        if not pending_tasks:
            return False
        subject, task = pending_tasks.popleft()
        tid = task["id"]
        await js.publish(subject, serialize_task(task))
        publish_times[tid] = time.time()
        return True

    async def _refill_inflight() -> None:
        """Publish pending tasks until the outstanding window is full.

        Outstanding = published-but-not-yet-resolved = len(publish_times).
        (publish_times entries are removed on result receipt and on DLQ sweep,
        so this naturally tracks in-flight work.)
        """
        while pending_tasks and len(publish_times) < max_inflight:
            await _publish_one()

    tracer = trace.get_tracer("slime.eval_coordinator")

    # Per-dataset ``{gym}/eval.step.<N>`` span context, captured at publish
    # time and reused in the collection loop below to parent each
    # ``validation_result`` span under its eval step. The eval.step span is
    # only *open* during publish (its ``with`` block closes before we start
    # collecting results), but a span's SpanContext stays valid after the
    # span ends — OTel/W&B nest by id, so a still-referenced parent context
    # is all we need to make the validation_result spans children rather
    # than detached roots that flood the trace tab.
    eval_step_ctx_by_dataset: dict[str, Any] = {}

    # Each gym gets its own ``{gym}/eval.step.<N>`` root span — W&B's trace
    # tab lists traces by root span name, and the ``{gym}/`` prefix groups
    # all of one gym's traces together (training + eval). The gym worker's
    # ``{gym}/task.<id>`` spans parent under this eval root via the injected
    # carrier (see the inject() call below), and the rollout subtree under
    # each ``task.<id>`` is identical to a training trajectory.
    for dataset_cfg in eval_datasets:
        gym_name = dataset_cfg.get("gym_name", "")
        if not gym_name:
            logger.warning("Eval dataset %s has no gym_name; skipping", dataset_cfg.get("name"))
            continue

        rows = _load_eval_dataset(dataset_cfg)
        n_samples = int(dataset_cfg.get("n_samples_per_eval_prompt") or 1)
        subject = f"{_EVAL_TASKS_SUBJECT_PREFIX}.{step}.{gym_name}"

        with tracer.start_as_current_span(
            f"{gym_name}/eval.step.{step}",
            attributes={
                "eval.step": str(step),
                "eval.kind": "training-eval",
                "eval.gym_name": gym_name,
                "eval.dataset": dataset_cfg["name"],
                "eval.n_rows": len(rows),
                "eval.n_samples_per_prompt": n_samples,
            },
        ) as eval_step_span:
            # Stash the eval.step span context so the result-collection loop
            # can parent each ``validation_result`` span under it (the span
            # itself ends when this block exits, but its context stays valid).
            eval_step_ctx_by_dataset[dataset_cfg["name"]] = trace.set_span_in_context(
                eval_step_span
            )
            # Inject the traceparent from the ``eval.step.<N>`` context ONCE
            # for the whole gym. We deliberately do NOT open a coordinator-
            # side per-row ``task.<id>`` span here: the gym dispatcher opens
            # its own ``{gym}/task.<id>`` span (AREnABase
            # GenerationDispatcher._process_task) and, when it sees this
            # carrier in ``task["trace_context"]``, parents that span under
            # this ``eval.step.<N>``. That makes the eval trace tree
            #
            #     {gym}/eval.step.<N>
            #       └─ {gym}/task.<id>          (gym-emitted)
            #            └─ rollout
            #                 ├─ rollout.setup
            #                 ├─ chat_completion / tool_call
            #                 └─ rollout.grade
            #
            # whose ``task.<id>`` subtree is byte-identical to a TRAINING
            # trajectory (training's gym pod opens the same ``{gym}/task.<id>``
            # root). If we opened a task span here too, the gym's span would
            # nest under it and produce a redundant double ``task.<id>`` layer
            # that no training trace has.
            # Inject the shared ``eval.step.<N>`` parent into every task so the
            # gym-emitted ``{gym}/task.<id>`` spans nest under one eval-step root
            # in the W&B trace UI. ALWAYS inject — including when local-file
            # tracing (ARENA_LOCAL_TRACES_DIR) is on.
            #
            # The earlier code SKIPPED the carrier under local tracing because
            # the old merge-on-export LocalDirSpanExporter merged every span
            # sharing the eval.step trace_id into ONE ``<trace_id>.json`` that
            # grew to GBs and was rewritten in full on each flush (O(n^2)). The
            # append-only per-process exporter (``<host>-<pid>.jsonl``, one
            # writer per process, never re-read) removes that failure mode: a
            # shared trace_id no longer means a shared file. So we no longer
            # trade away W&B eval-step grouping to keep local files bounded —
            # we get both. Spans re-stitch across shards on read (span ids are
            # globally unique) via ``merge_trace_shards`` / ``iter_traces``.
            carrier: dict[str, str] = {}
            inject(carrier)

            for task_idx, row in enumerate(rows):
                try:
                    # ONE task per prompt carrying ``n_samples``. The eval gym
                    # pod reads ``task["n_samples"]`` and runs the entire group
                    # in one pod (AsyncWorker fork pool), padding to exactly
                    # n_samples, exactly like the trainer's batched path.
                    # Previously we replicated each prompt into N ``-s<idx>``
                    # tasks scattered one-per-pod; now the whole group lands on
                    # one pod so the gym's group-metrics apply and the returned
                    # group is always complete.
                    task = _row_to_task(
                        row, dataset_cfg, task_idx, n_samples=n_samples
                    )
                except ValueError as e:
                    logger.warning("Skipping row %d: %s", task_idx, e)
                    continue
                tid = task["id"]  # base instance id, NO -s<idx> suffix
                task["trace_context"] = carrier
                task_registry[tid] = dataset_cfg["name"]
                # Enqueue rather than publish — the bounded-in-flight loop
                # below publishes up to `max_inflight` and tops up as
                # results return (so we never build a giant NATS backlog).
                # One enqueued task == one prompt-group (not one sample).
                pending_tasks.append((subject, task))
                total_tasks += 1

        logger.info(
            "Built %d eval tasks for %s (gym=%s, n_samples=%d, "
            "one task per prompt) -> %s",
            len(rows), dataset_cfg["name"], gym_name, n_samples, subject,
        )

    if total_tasks == 0:
        logger.warning("No eval tasks built.")
        await nc.close()
        return {}

    logger.info(
        "Total eval tasks built: %d. Publishing up to %d in-flight "
        "(gym_capacity=%d x headroom=%.1f); topping up as results return.",
        total_tasks, max_inflight, _gym_capacity, _inflight_headroom,
    )

    # Collect — no timeout, just keep pulling
    psub = await js.pull_subscribe(
        subject=results_subject,
        durable=consumer_name,
    )

    # Prime the pipe: publish the first window of tasks. The collection loop
    # tops it up as results arrive / tasks are swept.
    await _refill_inflight()

    # Seed every configured dataset with an empty list so a dataset where
    # every rollout failed still appears in the return dict (with zero
    # rewards). The trainer-side queue then writes a row at this eval/step
    # with avg_reward=0, n_samples=0 instead of skipping it — giving the
    # wandb chart a continuous x-axis with no gaps.
    # Rewards grouped by base_task_id, keyed by dataset name.
    # Structure: {dataset_name: {base_task_id: [reward_s0, reward_s1, ...]}}
    # Each inner list collects rewards from all n_samples attempts of the same task.
    # Flat reward list for avg_reward can be derived: [r for g in grouped.values() for r in g]
    results_by_dataset: dict[str, dict[str, list[float]]] = {
        d["name"]: {} for d in eval_datasets if d.get("name")
    }
    # Number of sample attempts per eval prompt, read from each dataset's config.
    # Used to determine group_size for pass@k and which pass@k values to report.
    n_samples_per_dataset: dict[str, int] = {
        d["name"]: int(d.get("n_samples_per_eval_prompt") or 1)
        for d in eval_datasets if d.get("name")
    }
    pass_threshold_per_dataset: dict[str, float] = {
        d["name"]: float(d.get("pass_threshold", 1.0))
        for d in eval_datasets if d.get("name")
    }
    collected = 0
    start_time = time.time()
    last_progress_log = start_time

    # DLQ sweep — mirrors the trainer-side mechanism added for G3.
    # Without this sweep the eval loop would spin on `while collected <
    # total_tasks` forever whenever a single prompt is a poison pill that
    # crashes every gym worker.
    #
    # The deadline is DERIVED from the eval-gym per-task ``timeout`` (the same
    # ``arena run --timeout`` value), mirroring the NATS-flow ordering in
    # AREnABase _run_setup.py: a task acks by its own timeout, and the gym
    # consumer's ack_wait = timeout + 300 (kill_deadline+200), so NATS never
    # redelivers a still-running task. The coordinator deadline only needs to
    # sit just ABOVE that ack_wait as a backstop for a truly-hung worker:
    #   deadline = max_gym_timeout + 300 (ack_wait slack) + 200 (margin).
    # (Previously a stale constant 3*3600+1800=12600s — built on the wrong
    # assumption that no --timeout was passed, so a task ran the full
    # ack_wait x max_deliver. That killed most of the val set in exp16.)
    # Still overridable via EVAL_TASK_DEADLINE_SECS. The per-task timeout is
    # the top-level ``eval_timeout`` (EVAL_TIMEOUT env, source of truth), with
    # any per-gym ``timeout`` override taking precedence — mirror that here so
    # the deadline tracks the largest timeout actually in effect.
    try:
        # ``or "3000"`` so a PRESENT-but-EMPTY env var (template renders
        # ``value: "${EVAL_TIMEOUT}"`` to "" when deploy.sh doesn't export it)
        # falls back instead of forcing the except branch below.
        _eval_timeout_default = int(os.environ.get("EVAL_TIMEOUT") or "3000")
        _eval_gyms_cfg = json.loads(os.environ.get("EVAL_GYMS_CONFIG", "[]") or "[]")
        _max_gym_timeout = max(
            (int(g.get("timeout", _eval_timeout_default)) for g in _eval_gyms_cfg),
            default=_eval_timeout_default,
        )
    except (ValueError, TypeError):
        _max_gym_timeout = 3000
    eval_task_deadline_secs = float(
        os.environ.get("EVAL_TASK_DEADLINE_SECS", "") or (_max_gym_timeout + 300 + 200)
    )
    eval_dlq_sweep_interval_secs = 30.0
    last_eval_dlq_sweep = time.time()

    # NATS-py raises ServiceUnavailableError (503) when the durable consumer
    # backing the pull-subscription disappears (e.g. NATS pod was rescheduled
    # and lost JetStream state). Re-create the consumer + psub on the fly
    # rather than hanging forever on retried fetches.
    from miles_plugins.arena.nats_arena.nats_rollout import _is_nats_reconnect_error  # local import — avoids cycles

    async def _reopen_psub():
        nonlocal nc, js, psub
        backoff = 1.0
        while True:
            try:
                # Re-ensure streams (idempotent — only creates if missing).
                await _ensure_nats_streams(nats_url, step)
                if nc.is_closed:
                    nc = await nats.connect(
                        nats_url, max_reconnect_attempts=-1, reconnect_time_wait=2,
                    )
                    js = nc.jetstream()
                # Recreate the durable eval consumer if missing.
                try:
                    await js.consumer_info(_EVAL_RESULTS_STREAM, consumer_name)
                except Exception:
                    from nats.js.api import ConsumerConfig
                    await js.add_consumer(
                        stream=_EVAL_RESULTS_STREAM,
                        config=ConsumerConfig(
                            durable_name=consumer_name,
                            filter_subject=results_subject,
                            ack_policy="explicit",
                            max_deliver=3,
                            ack_wait=3600,
                        ),
                    )
                    logger.info("Recreated missing eval consumer %s", consumer_name)
                psub = await js.pull_subscribe(
                    subject=results_subject,
                    durable=consumer_name,
                )
                logger.info(
                    "Eval pull-subscription reopened (consumer=%s)", consumer_name,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Eval reconnect failed: %s — retrying in %.1fs", exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _handle_result_msg(result: dict) -> bool:
        """Process one parsed result. Returns True if it was a fresh (non-dup)
        result that incremented ``collected``. Mutates collected / seen_task_ids
        / results_by_dataset / publish_times via nonlocal/closure."""
        nonlocal collected
        tid = result.get("task_id", "")

        # Dedup + DLQ-override, three cases:
        #   (a) tid in dlq_swept  -> the sweep counted it as a synthetic miss,
        #       but a REAL result has now arrived. Record its reward to correct
        #       the metrics; do NOT re-increment collected (already counted).
        #   (b) tid in seen_task_ids (and not swept) -> genuine duplicate
        #       (NATS redelivery / consumer replay). Ack and skip.
        #   (c) first arrival -> normal path, increments collected.
        is_override = tid in dlq_swept
        if is_override:
            dlq_swept.discard(tid)
            logger.info(
                "Eval: late real result for DLQ-swept task_id=%s — overriding "
                "the synthetic missing-result with its actual reward.", tid,
            )
        elif tid in seen_task_ids:
            logger.debug(
                "Eval: duplicate result for task_id=%s — already counted; "
                "ack and skip.", tid,
            )
            return False
        else:
            seen_task_ids.add(tid)

        dataset_name = task_registry.get(tid)
        if dataset_name is None:
            base_id = tid.rsplit("-s", 1)[0]
            dataset_name = task_registry.get(base_id, "unknown")

        trajectories = extract_trajectories(result)
        # One result now carries the WHOLE group of n_samples trajectories
        # (the gym pod fanned them out and padded to exactly n_samples), so
        # ``tid`` is the plain instance id — no ``-s<idx>`` suffix to strip.
        # All n_samples rewards accumulate under this single tid.
        base_tid = tid
        for sample_idx, traj in enumerate(trajectories):
            reward = traj.get("reward", 0.0)
            results_by_dataset.setdefault(dataset_name, {}).setdefault(base_tid, []).append(reward)

            # Emit a trace span for each completion result (appears in Weave Evaluation dashboard).
            #
            # Parent under this dataset's ``eval.step.<N>`` span (captured
            # at publish time) so validation_result spans nest INSIDE the
            # eval step instead of appearing as separate root spans that
            # bury eval.step.<N> in the W&B trace tab. ``context=None``
            # would fall back to the (empty) current context → detached
            # root, which is the old buggy behavior. The trajectory itself
            # is NOT duplicated onto this span — the full conversation
            # already lives in the gym-side ``rollout`` / ``chat_completion``
            # spans under ``{gym}/task.<id>`` in the same eval.step tree.
            #
            # Suffix the span name with ``.s<sample_idx>`` so the n_samples
            # trajectories — which now all share one ``tid`` — get distinct
            # span names instead of N colliding ``...step.<N>.<tid>`` spans.
            parent_ctx = eval_step_ctx_by_dataset.get(dataset_name)
            with tracer.start_as_current_span(
                f"validation_result.step.{step}.{tid}.s{sample_idx}",
                context=parent_ctx,
                attributes={
                    "eval.step": step,
                    "eval.task_id": tid,
                    "eval.base_task_id": base_tid,
                    "eval.sample_index": sample_idx,
                    "eval.dataset": dataset_name,
                    "eval.reward": reward,
                    "eval.pass": reward >= pass_threshold_per_dataset.get(dataset_name, 1.0),
                },
            ):
                pass

        publish_times.pop(tid, None)
        if is_override:
            # Reward corrected in place; this tid was already counted by the
            # DLQ sweep, so do NOT double-count collected.
            return False
        collected += 1
        return True

    async def _drain_results(max_batches: int = 200) -> int:
        """Fetch + process ALL currently-available result msgs until the stream
        yields nothing (a fetch times out empty). Returns fresh results counted.

        Critical for the DLQ sweep: a result can be sitting in the stream,
        already published by the gym but NOT YET FETCHED by this consumer,
        while the task's publish-relative deadline expires. Without draining
        first, the sweep marks that task ``missing-result`` and bumps
        ``collected`` to total_tasks, so the outer loop EXITS before ever
        fetching the real (often reward=1.0) result — silently dropping a
        successful trajectory. Draining before sweeping closes that race.
        """
        nonlocal psub
        fresh = 0
        for _ in range(max_batches):
            try:
                msgs = await psub.fetch(batch=50, timeout=_RESULT_POLL_INTERVAL)
            except TimeoutError:
                break  # stream drained — nothing left to fetch right now
            except Exception as exc:
                if _is_nats_reconnect_error(exc):
                    await _reopen_psub()
                    continue
                logger.warning("drain fetch error: %s", exc)
                break
            if not msgs:
                break
            for msg in msgs:
                try:
                    result = parse_result(msg.data)
                    await msg.ack()
                except Exception as e:
                    logger.warning("Failed to parse eval result: %s", e)
                    await msg.ack()
                    continue
                if _handle_result_msg(result):
                    fresh += 1
        return fresh

    while collected < total_tasks:
        try:
            msgs = await psub.fetch(
                batch=min(50, total_tasks - collected),
                timeout=_RESULT_POLL_INTERVAL,
            )
        except TimeoutError:
            msgs = []
        except Exception as exc:
            if _is_nats_reconnect_error(exc):
                logger.warning(
                    "Eval fetch hit a reconnect-recoverable error (%s) — "
                    "reopening pull-subscription.", exc,
                )
                await _reopen_psub()
                continue
            logger.warning("fetch error: %s", exc)
            await asyncio.sleep(2)
            continue

        for msg in msgs:
            try:
                result = parse_result(msg.data)
                await msg.ack()
            except Exception as e:
                logger.warning("Failed to parse eval result: %s", e)
                await msg.ack()
                continue
            _handle_result_msg(result)

        # A result freed an in-flight slot (publish_times shrank) — top up
        # the window with the next pending task(s).
        await _refill_inflight()

        now = time.time()
        if now - last_progress_log >= _PROGRESS_LOG_INTERVAL:
            logger.info(
                "Eval progress (step=%s): %d/%d results (%.0fs elapsed, "
                "in_flight=%d/%d, pending=%d)",
                step, collected, total_tasks, now - start_time,
                len(publish_times), max_inflight, len(pending_tasks),
            )
            last_progress_log = now

        # DLQ sweep: drop tasks whose retry budget is exhausted so the
        # outer `while collected < total_tasks` loop can complete.
        # We bump `collected` in place of the missing result so the
        # progress check converges; the corresponding entries in
        # `results_by_dataset` simply have one fewer reward (zero or
        # absent — the eval just gets a slightly smaller n_samples
        # for that prompt). A WARNING per expired task surfaces the
        # offending instance_ids for human inspection.
        if now - last_eval_dlq_sweep >= eval_dlq_sweep_interval_secs:
            last_eval_dlq_sweep = now
            expired = [
                t for t, ts in publish_times.items()
                if now - ts > eval_task_deadline_secs
            ]
            if expired:
                # RACE GUARD: a task's result may already be sitting in the
                # results stream, published by the gym but not yet fetched by
                # this consumer, even though its publish-relative deadline has
                # passed. Drain everything available FIRST, then recompute
                # ``expired`` — only tasks whose result is genuinely still
                # absent get swept. Without this, the sweep bumps ``collected``
                # to total_tasks and the outer loop exits before fetching the
                # real (often reward=1.0) result, silently dropping a
                # successful trajectory (observed: num_pending=4 at exit,
                # 5 "missing" prompts that had actually solved).
                drained = await _drain_results()
                if drained:
                    logger.info(
                        "Eval DLQ pre-sweep drain recovered %d result(s); "
                        "re-checking expiry.", drained,
                    )
                now = time.time()
                expired = [
                    t for t, ts in publish_times.items()
                    if now - ts > eval_task_deadline_secs
                ]
            if expired:
                for tid in expired:
                    age = now - publish_times.pop(tid, now)
                    logger.warning(
                        "Eval DLQ: task %s exceeded deadline (%.0fs > %.0fs) — "
                        "NATS retry budget likely exhausted. Marking as "
                        "missing-result so eval can complete.",
                        tid, age, eval_task_deadline_secs,
                    )
                    seen_task_ids.add(tid)
                    dlq_swept.add(tid)  # allow a late real result to override
                    collected += 1
                logger.warning(
                    "Eval DLQ swept %d expired tasks (collected now %d/%d).",
                    len(expired), collected, total_tasks,
                )
                # Swept tasks freed in-flight slots — top up the window.
                await _refill_inflight()

    # Final drain: the loop may have exited via DLQ-sweep bumping ``collected``
    # to total_tasks while late/redelivered results were still unfetched in the
    # stream. Pull whatever remains so those (often successful) results replace
    # the synthetic missing-result slots before we compute metrics. Results for
    # already-counted tids are deduped; genuinely-new ones correct the reward.
    final_recovered = await _drain_results()
    if final_recovered:
        logger.warning(
            "Eval post-loop drain recovered %d result(s) that had been marked "
            "missing by the DLQ sweep — metrics corrected.", final_recovered,
        )

    elapsed = time.time() - start_time
    logger.info("Collected all %d/%d results in %.0fs", collected, total_tasks, elapsed)

    await nc.close()

    return {
        name: {
            "rewards_grouped": grouped,
            "count": sum(len(v) for v in grouped.values()),
            "n_samples_per_eval_prompt": n_samples_per_dataset.get(name, 1),
            "pass_threshold": pass_threshold_per_dataset.get(name, 1.0),
        }
        for name, grouped in results_by_dataset.items()
    }


# ---------------------------------------------------------------------------
# wandb logging
# ---------------------------------------------------------------------------

# File-queue protocol for eval metrics:
#
#   <EVAL_METRICS_DIR>/
#     step_<N>.json          — one per completed eval, written atomically
#     .trainer_alive         — heartbeat touched by the trainer; presence means
#                              the trainer is the wandb writer, coordinator
#                              just drops its file and exits.
#     .flush_lock            — held by whichever coordinator is doing the
#                              post-training flush. O_EXCL acquire.
#
# The single-writer invariant (only one process calls wandb.log at a time on a
# given run) is what keeps train/rollout/perf step axes from getting clobbered.

_TRAINER_ALIVE_NAME = ".trainer_alive"
_FLUSH_LOCK_NAME = ".flush_lock"
_TRAINER_HEARTBEAT_STALE_SECONDS = 600   # 10 min without heartbeat → trainer dead
_FLUSH_LOCK_STALE_SECONDS = 1800         # 30 min holding lock → break it


def _estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimate pass@k using the unbiased estimator from the Codex paper."""
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def _compute_pass_at_k(
    rewards_grouped: dict[str, list[float]],
    group_size: int,
    threshold: float = 1.0,
) -> dict[str, float]:
    """Compute pass@k metrics from rewards grouped by task_id.

    Args:
        rewards_grouped: {base_task_id: [reward_s0, reward_s1, ...]}
        group_size: expected number of samples per task
        threshold: reward >= threshold counts as "pass"

    Only includes tasks that have exactly group_size samples (incomplete
    groups are excluded from the computation).
    """
    if group_size <= 1:
        return {}

    # Filter to groups with complete samples
    complete_groups = [
        rewards for rewards in rewards_grouped.values()
        if len(rewards) == group_size
    ]
    if not complete_groups:
        return {}

    metrics: dict[str, float] = {}
    pass_k_values = [2**i for i in range(int(math.log2(group_size)) + 1)]

    for k in pass_k_values:
        estimates = []
        for group_rewards in complete_groups:
            num_correct = sum(1 for r in group_rewards if r >= threshold)
            estimates.append(_estimate_pass_at_k(group_size, num_correct, k))
        metrics[f"pass@{k}"] = sum(estimates) / len(estimates)

    metrics["n_complete_groups"] = float(len(complete_groups))
    return metrics


def _build_metrics_payload(step: int, results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Build the wandb-shaped payload for one eval step.

    Always emits ``eval/<dataset>/avg_reward`` and ``.../n_samples`` for
    every configured dataset, even when no rollouts produced a reward
    (e.g. every rollout in this dataset hit a context-length 502).
    Empty datasets land as ``avg_reward=0.0, n_samples=0`` so the wandb
    chart has a continuous row at this ``eval/step`` instead of a gap
    that would visually shift later evals to the wrong x-position.

    When n_samples_per_eval_prompt > 1, also emits pass@k metrics.
    """
    payload: dict[str, float] = {"eval/step": step}
    for name, data in results.items():
        rewards_grouped = data.get("rewards_grouped", {})
        rewards = [r for group in rewards_grouped.values() for r in group]
        threshold = data.get("pass_threshold", 1.0)
        if rewards:
            n = len(rewards)
            avg = sum(rewards) / n
            binary_reward = sum(1 for r in rewards if r >= threshold) / n
        else:
            n, avg, binary_reward = 0, 0.0, 0.0
        payload[f"eval/{name}/avg_reward"] = avg
        payload[f"eval/{name}/n_samples"] = n
        payload[f"eval/{name}/binary_reward"] = binary_reward

        # Compute pass@k if we have grouped samples
        group_size = data.get("n_samples_per_eval_prompt", 1)
        if group_size > 1 and rewards_grouped:
            pass_k_metrics = _compute_pass_at_k(rewards_grouped, group_size, threshold=threshold)
            for metric_name, value in pass_k_metrics.items():
                payload[f"eval/{name}/{metric_name}"] = value
            logger.info(
                "Eval %s @ step=%d: avg_reward=%.4f binary_reward=%.4f n=%d pass@1=%.4f groups=%d (threshold=%.2f)",
                name, step, avg, binary_reward, n,
                pass_k_metrics.get("pass@1", 0.0),
                int(pass_k_metrics.get("n_complete_groups", 0)),
                threshold,
            )
        else:
            logger.info(
                "Eval %s @ step=%d: avg_reward=%.4f binary_reward=%.4f n=%d (threshold=%.2f)",
                name, step, avg, binary_reward, n, threshold,
            )
    return payload


def _atomic_write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.rename(tmp, path)  # atomic on POSIX


def _trainer_is_alive(metrics_dir: str) -> bool:
    sentinel = os.path.join(metrics_dir, _TRAINER_ALIVE_NAME)
    if not os.path.exists(sentinel):
        return False
    age = time.time() - os.path.getmtime(sentinel)
    return age < _TRAINER_HEARTBEAT_STALE_SECONDS


def _try_acquire_flush_lock(metrics_dir: str) -> bool:
    lock = os.path.join(metrics_dir, _FLUSH_LOCK_NAME)
    # Break a stale lock (process crashed mid-flush).
    if os.path.exists(lock):
        try:
            if time.time() - os.path.getmtime(lock) > _FLUSH_LOCK_STALE_SECONDS:
                logger.warning("Removing stale flush lock at %s", lock)
                os.unlink(lock)
        except FileNotFoundError:
            pass
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_flush_lock(metrics_dir: str) -> None:
    try:
        os.unlink(os.path.join(metrics_dir, _FLUSH_LOCK_NAME))
    except FileNotFoundError:
        pass


def _flush_pending_to_wandb(metrics_dir: str) -> None:
    """Resume the trainer's wandb run and log every queued step_*.json file.

    Only call this when the trainer is no longer running. Holding `.flush_lock`
    is the caller's responsibility.
    """
    run_id = os.environ.get("WANDB_RUN_ID", "")
    project = os.environ.get("WANDB_PROJECT", "")
    if not run_id or not project:
        logger.warning("WANDB_RUN_ID/WANDB_PROJECT missing; cannot flush %s", metrics_dir)
        return
    # Entity/team the trainer created the run under. The trainer inits its
    # wandb run with entity=WANDB_TEAM (e.g. ``arena``); without passing the
    # SAME entity here, wandb.init resolves to the API key's DEFAULT entity —
    # the user's PERSONAL one (e.g. ``rahsubb``) — and ``resume="must"`` then
    # looks for ``rahsubb/<project>/<run_id>``, which doesn't exist, so the
    # flush fails with "run has not been initialized" and eval metrics are
    # lost. Mirror the trainer's entity resolution so the flush targets the
    # SAME run. Empty => let wandb use its default (legacy behavior).
    entity = os.environ.get("WANDB_TEAM", "") or os.environ.get("WANDB_ENTITY", "")
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; cannot flush %s", metrics_dir)
        return

    pending = sorted(
        f for f in os.listdir(metrics_dir)
        if f.startswith("step_") and f.endswith(".json")
    )
    if not pending:
        return

    _init_kwargs: dict[str, Any] = dict(
        id=run_id,
        project=project,
        settings=wandb.Settings(
            base_url=os.environ.get("WANDB_BASE_URL", ""),
            _disable_stats=True,
            _disable_meta=True,
        ),
    )
    if entity:
        _init_kwargs["entity"] = entity
    try:
        # resume="must" — attach to the trainer's existing run.
        wandb.init(resume="must", **_init_kwargs)
    except Exception as e:
        # The run may not exist under this entity yet (e.g. trainer died
        # before ever creating it, or an entity mismatch we can't resolve).
        # Fall back to resume="allow" so eval metrics still land in a run
        # rather than being dropped entirely.
        logger.warning(
            "wandb.init(resume='must', entity=%r) failed (%s); retrying with "
            "resume='allow' so eval metrics aren't lost.", entity or "<default>", e,
        )
        try:
            wandb.init(resume="allow", **_init_kwargs)
        except Exception as e2:
            logger.error("wandb.init for post-training flush failed: %s", e2)
            return

    try:
        # ``rollout/*`` etc. live in the trainer; here we only need eval. wandb's
        # ``define_metric`` glob is **suffix-only** (one trailing ``*``,
        # matching a single path segment) so a 2-segment pattern like
        # ``eval/*/*`` is rejected. We register the broad ``eval/*`` for
        # any single-segment keys, then per-dataset ``eval/<name>/*``
        # patterns for the nested keys we're about to log
        # (``eval/<dataset>/avg_reward``, ``.../n_samples``).
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
    except Exception as e:
        logger.warning("define_metric failed (continuing): %s", e)

    flushed = 0
    declared_datasets: set[str] = set()
    for fname in pending:
        fpath = os.path.join(metrics_dir, fname)
        try:
            payload = json.load(open(fpath))
        except Exception as e:
            logger.error("Failed to read %s: %s; skipping", fpath, e)
            continue
        # Declare per-dataset step_metric bindings as soon as we see the
        # dataset name in a payload — wandb's define_metric is idempotent
        # for repeats and we don't have the dataset list otherwise.
        for key in payload:
            if not key.startswith("eval/") or key == "eval/step":
                continue
            parts = key.split("/")
            if len(parts) >= 3:
                name = parts[1]
                if name not in declared_datasets:
                    try:
                        wandb.define_metric(f"eval/{name}/*", step_metric="eval/step")
                        declared_datasets.add(name)
                    except Exception as e:
                        logger.warning(
                            "define_metric failed for eval/%s/* (continuing): %s",
                            name, e,
                        )
                        declared_datasets.add(name)
        try:
            wandb.log(payload)
            os.unlink(fpath)
            flushed += 1
        except Exception as e:
            logger.error("wandb.log failed for %s: %s", fpath, e)

    logger.info("Post-training flush wrote %d eval result(s) to wandb", flushed)
    try:
        wandb.finish()
    except Exception:
        pass


def _persist_eval_metrics(step: int, results: dict[str, dict[str, Any]]) -> None:
    """Write this eval's metrics to the shared queue and (optionally) flush.

    Normal case (trainer alive): write step_<N>.json and exit. Trainer drains
    the dir at the top of every train iteration.

    Edge case (trainer already exited): acquire the flush lock and drain the
    whole dir ourselves so nothing is lost.
    """
    payload = _build_metrics_payload(step, results)
    # Only skip when there's truly nothing to record — i.e. no datasets
    # were configured for this run (payload only has ``eval/step``).
    # When datasets ARE configured but every rollout failed, the payload
    # still carries ``avg_reward=0, n_samples=0`` per dataset and we MUST
    # write the file so wandb sees a row at this ``eval/step``. Skipping
    # creates gaps in the chart that visually shift later evals (e.g. if
    # step 2 fails entirely, step 3's row is plotted at x=3 but the chart
    # reads it as "the next eval after step 1").
    if len(payload) <= 1:
        logger.info(
            "No datasets configured for step=%d; nothing to persist", step,
        )
        return

    metrics_dir = os.environ.get("EVAL_METRICS_DIR", "")
    if not metrics_dir:
        logger.warning("EVAL_METRICS_DIR not set; eval metrics for step=%d will be lost", step)
        return

    os.makedirs(metrics_dir, exist_ok=True)
    fpath = os.path.join(metrics_dir, f"step_{step:08d}.json")
    _atomic_write_json(fpath, payload)
    logger.info("Persisted %d eval metrics to %s", len(payload) - 1, fpath)

    if _trainer_is_alive(metrics_dir):
        return  # trainer will drain on its next iteration

    # Trainer has exited (or its heartbeat is stale). Last coordinator standing
    # flushes the queue. Use a lockfile so concurrent late coordinators
    # serialize.
    if not _try_acquire_flush_lock(metrics_dir):
        logger.info(
            "Trainer not alive but another coordinator holds the flush lock; "
            "leaving %s for them to pick up", fpath,
        )
        return

    try:
        _flush_pending_to_wandb(metrics_dir)
    finally:
        _release_flush_lock(metrics_dir)


# ---------------------------------------------------------------------------
# HaLLMark integration
# ---------------------------------------------------------------------------

# TODO(temporary workaround): HaLLMark currently requires --sandbox hallmark-k8s
# for per-sample isolated benchmarks (Terminal-Bench, Mini-SWE variants) but this
# flag breaks task-scoped HTTP service benchmarks (tau2, BFCL). We work around
# this by splitting benchmarks into separate requests. Remove once HaLLMark
# handles sandbox selection per-benchmark automatically.
#
# Prefix-based eval_args for HaLLMark benchmarks. Matched in order — first
# prefix that matches wins.
_HALLMARK_BENCHMARK_EVAL_ARGS_PREFIXES: list[tuple[str, str]] = [
    ("amzn_hallmark/terminal_bench", "--sandbox hallmark-k8s"),
    ("amzn_hallmark/mini_swe_agent", "--sandbox hallmark-k8s"),
]


def _get_hallmark_eval_args(benchmark: str) -> str:
    for prefix, eval_args in _HALLMARK_BENCHMARK_EVAL_ARGS_PREFIXES:
        if benchmark.startswith(prefix):
            return eval_args
    return ""


def _upload_hf_checkpoint_to_s3(hf_path: str, step: int) -> str:
    """Upload HF checkpoint to S3 for HaLLMark to consume.

    Uploads to: s3://<bucket>/<prefix>/<job_name>/step_<step>/
    Returns the S3 URI of the uploaded checkpoint.
    """
    import boto3
    from pathlib import Path

    bucket = os.environ.get("HALLMARK_S3_BUCKET", "arena-model-eval-ap-south-1")
    user_alias = os.environ.get("USER_ALIAS", "unknown")
    prefix = f"{user_alias}-sandbox"
    job_name = os.environ.get("JOB_NAME", "unknown")

    s3_key_prefix = f"{prefix}/{job_name}/step_{step}"
    s3_uri = f"s3://{bucket}/{s3_key_prefix}/"
    logger.info("Uploading HF checkpoint to S3: %s -> %s", hf_path, s3_uri)

    s3 = boto3.client("s3")
    uploaded = 0

    if hf_path.startswith("s3://"):
        # Checkpoint already lives in S3 (--save-hf points at s3://). Copy
        # S3 -> S3 into the HaLLMark bucket/layout instead of re-uploading
        # from a local dir that doesn't exist on this pod.
        src_rest = hf_path[len("s3://"):]
        src_bucket, _, src_prefix = src_rest.partition("/")
        src_prefix = src_prefix.strip("/")
        list_prefix = f"{src_prefix}/" if src_prefix else ""
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=src_bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(list_prefix):]
                if not relative:
                    continue
                dst_key = f"{s3_key_prefix}/{relative}"
                try:
                    s3.copy({"Bucket": src_bucket, "Key": key}, bucket, dst_key)
                    uploaded += 1
                except Exception as e:
                    logger.warning("Failed to copy %s: %s", relative, e)
    else:
        local_root = Path(hf_path)
        for file_path in local_root.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_root)
            s3_key = f"{s3_key_prefix}/{relative}"
            try:
                s3.upload_file(str(file_path), bucket, s3_key)
                uploaded += 1
            except Exception as e:
                logger.warning("Failed to upload %s: %s", relative, e)

    if uploaded == 0:
        raise RuntimeError(f"No files found to upload in {hf_path}")

    logger.info("HF checkpoint uploaded to %s (%d files)", s3_uri, uploaded)
    return s3_uri


def _run_hallmark_benchmarks(benchmarks: list[str], step: int, checkpoint_s3_path: str = "") -> list[dict[str, Any]]:
    """Trigger HaLLMark benchmarks via the HaLLMark API.

    Benchmarks are split into groups based on sandbox requirements:
    - Per-sample k8s sandbox (terminal_bench, mini_swe_agent*): sent with eval_args="--sandbox hallmark-k8s"
    - Task-scoped HTTP service (tau2, bfcl, etc.): sent without eval_args

    Each non-empty group becomes a separate API request. Returns a list of
    per-group API response dicts (each has "name"/"argo_ui_url" on success,
    or "error" on failure).
    """
    import boto3
    import requests
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    hallmark_api_url = os.environ.get(
        "HALLMARK_API_URL", "https://jhekgocqk3.execute-api.us-west-2.amazonaws.com/v1/workflows"
    )
    model_profile = os.environ.get("HALLMARK_MODEL_PROFILE", "qwen3.5-9b-rl-gym")

    if not checkpoint_s3_path:
        bucket = os.environ.get("HALLMARK_S3_BUCKET", "arena-model-eval-ap-south-1")
        user_alias = os.environ.get("USER_ALIAS", "unknown")
        job_name = os.environ.get("JOB_NAME", "unknown")
        checkpoint_s3_path = f"s3://{bucket}/{user_alias}-sandbox/{job_name}/step_{step}/"

    # Group benchmarks by their eval_args so each distinct set of args gets one request.
    groups_by_args: dict[str, list[str]] = {}
    for b in benchmarks:
        args = _get_hallmark_eval_args(b)
        groups_by_args.setdefault(args, []).append(b)

    session = boto3.Session(region_name=os.environ.get("HALLMARK_API_REGION", "us-west-2"))
    credentials = session.get_credentials().get_frozen_credentials()
    user_alias = os.environ.get("USER_ALIAS", "")
    metrics_dir = os.environ.get("EVAL_METRICS_DIR", "")
    results: list[dict[str, Any]] = []

    for eval_args, group_benchmarks in groups_by_args.items():
        group_label = re.sub(r"[^a-z0-9]+", "_", eval_args.strip().lower()).strip("_") if eval_args else "standard"
        payload: dict[str, Any] = {
            "template": "hallmark-hosted-vllm-inference",
            "checkpoint": checkpoint_s3_path,
            "model_profile": model_profile,
            "tasks": " ".join(group_benchmarks),
        }
        if eval_args:
            payload["eval_args"] = eval_args

        logger.info(
            "HaLLMark API call (step=%d, group=%s, benchmarks=%s): POST %s payload=%s",
            step, group_label, group_benchmarks, hallmark_api_url, json.dumps(payload),
        )

        try:
            data = json.dumps(payload)
            headers = {"Content-Type": "application/json", "X-Forwarded-User": user_alias}
            aws_request = AWSRequest(method="POST", url=hallmark_api_url, data=data, headers=headers)
            SigV4Auth(credentials, "execute-api", session.region_name).add_auth(aws_request)

            resp = requests.post(hallmark_api_url, headers=dict(aws_request.headers), data=data, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "HaLLMark workflow submitted (step=%d, group=%s): name=%s, argo_ui_url=%s",
                step, group_label, result.get("name"), result.get("argo_ui_url"),
            )
            results.append(result)
        except Exception as e:
            logger.error("HaLLMark API call failed (step=%d, group=%s): %s", step, group_label, e)
            results.append({"error": str(e)})
            continue

        # Persist to metrics dir for trainer to drain to wandb.
        try:
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
                hallmark_meta = {
                    "eval/step": step,
                    f"eval/hallmark/{group_label}/argo_ui_url": result.get("argo_ui_url", ""),
                    f"eval/hallmark/{group_label}/checkpoint": checkpoint_s3_path,
                }
                fpath = os.path.join(metrics_dir, f"step_{step:08d}_hallmark_{group_label}.json")
                _atomic_write_json(fpath, hallmark_meta)
        except Exception as e:
            logger.warning("Failed to persist HaLLMark metadata (step=%d, group=%s): %s", step, group_label, e)

    return results






# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    job_name = os.environ["JOB_NAME"]
    namespace = os.environ["NAMESPACE"]
    step = os.environ["EVAL_STEP"]
    coord_job_name = f"{job_name}-eval-coord-step-{step}"
    nats_url = os.environ.get("NATS_URL", f"nats://{os.environ.get('NATS_SVC', 'nats')}:4222")

    logger.info("Eval coordinator starting: job=%s step=%s ns=%s", job_name, step, namespace)

    # 0. Install OTLP tracer so the eval.step.<N> + per-row task.<id> spans
    # we open in _publish_and_collect actually export to W&B. Without this
    # the global provider is the default ProxyTracerProvider and inject()
    # yields an empty carrier — gym worker rollouts would then be detached
    # roots instead of children of the eval trace.
    _install_otlp_tracer_provider()

    # 1. Resolve own Job UID for OwnerReferences
    k8s = _load_k8s()
    owner_ref = _get_own_job_owner_ref(k8s, namespace, coord_job_name)
    logger.info("Own Job OwnerRef resolved: uid=%s", owner_ref["uid"])

    # 2. Ensure NATS streams (idempotent)
    asyncio.run(_ensure_nats_streams(nats_url, step))

    # 3. Build env vars for template rendering
    env: dict[str, str] = {k: os.environ.get(k, "") for k in os.environ}
    env["STEP"] = step  # alias for templates that use ${STEP}

    templates_dir = _find_templates_dir()
    logger.info("Using templates dir: %s", templates_dir)

    # 4. Render and apply eval-sglang Deployment + Service
    sglang_template = (templates_dir / "eval-sglang-server.yaml").read_text()
    sglang_docs = _split_yaml_docs(_render(sglang_template, env))
    _apply_with_owner(k8s, sglang_docs, namespace, owner_ref)

    # 5. Render and apply eval-gym-worker Deployments (one per gym)
    gym_template = (templates_dir / "eval-gym-workers.yaml").read_text()
    eval_gyms_raw = os.environ.get("EVAL_GYMS_CONFIG", "[]") or "[]"
    eval_gyms = json.loads(eval_gyms_raw) if isinstance(eval_gyms_raw, str) else eval_gyms_raw

    for gym_cfg in eval_gyms:
        gym_name = gym_cfg["gym_name"]
        gym_env = dict(env)
        gym_env["GYM_NAME"] = gym_name
        gym_env["GYM_SLUG"] = gym_name.replace("_", "-")
        gym_env["EVAL_REPLICAS"] = str(gym_cfg.get("replicas", 4))
        gym_env["EVAL_GYM_CONCURRENCY"] = str(gym_cfg.get("concurrency", 1))
        gym_env["EVAL_MAX_ITERATIONS"] = str(gym_cfg.get("max_iterations", 1000))
        # Per-task wall-clock passed to ``arena run --timeout`` so an eval task
        # is killed by its OWN timeout before NATS redelivers it. Without this
        # the eval-gym ran ``arena run`` with no --timeout, so a long task ran
        # the full ack_wait x max_deliver (~3h) before the DLQ deadline marked
        # it missing-result — killing the bulk of the val set (exp16: 285/352).
        #
        # SOURCE OF TRUTH = the top-level ``eval_timeout`` config (propagated
        # here as the EVAL_TIMEOUT env by eval_rollout._build_coordinator_env).
        # The per-gym ``eval-gyms.yaml`` ``timeout`` is an OPTIONAL override —
        # only used when explicitly set on that gym. Stays < the gym consumer
        # ack_wait (timeout+300, derived in AREnABase _run_setup) so a task
        # acks by its own timeout before redelivery.
        _eval_timeout_default = os.environ.get("EVAL_TIMEOUT", "3000")
        gym_env["EVAL_INSTANCE_TIMEOUT"] = str(gym_cfg.get("timeout", _eval_timeout_default))
        gym_env["EVAL_GYM_CPU"] = str(gym_cfg.get("cpu", "4"))
        gym_env["EVAL_GYM_MEMORY"] = str(gym_cfg.get("memory", "80Gi"))
        gym_env["EVAL_GYM_EPHEMERAL_STORAGE"] = str(gym_cfg.get("ephemeral_storage", "100Gi"))
        if gym_cfg.get("image"):
            gym_env["ARENA_IMAGE"] = gym_cfg["image"]
        gym_docs = _split_yaml_docs(_render(gym_template, gym_env))
        _apply_with_owner(k8s, gym_docs, namespace, owner_ref)
        logger.info("Applied eval gym worker: gym=%s replicas=%s", gym_name, gym_env["EVAL_REPLICAS"])

    # 6. Wait for SGLang readiness
    # Wrap everything from here in try/finally so eval infra is always cleaned up,
    # even on SGLang timeout or other failures.
    sglang_svc = f"{job_name}-eval-sglang-svc-step-{step}"
    try:
        _wait_for_sglang_ready(sglang_svc, namespace)

        # 7. Run NATS gym eval + HaLLMark benchmarks in parallel
        import concurrent.futures

        # 7a. Parse eval configs
        eval_datasets_raw = os.environ.get("EVAL_DATASETS_JSON", "[]") or "[]"
        eval_datasets = json.loads(eval_datasets_raw) if isinstance(eval_datasets_raw, str) else eval_datasets_raw

        hallmark_enabled = os.environ.get("HALLMARK_ENABLED", "0") == "1"
        hallmark_benchmarks_raw = os.environ.get("HALLMARK_BENCHMARKS", "")
        hallmark_benchmarks = [b.strip() for b in hallmark_benchmarks_raw.split(",") if b.strip()] if hallmark_enabled else []

        hf_path = os.environ.get("EVAL_HF_PATH", "")

        if not eval_datasets and not hallmark_benchmarks:
            logger.warning("No eval datasets or HaLLMark benchmarks configured. Exiting.")
            return 0

        if hallmark_benchmarks and not hf_path:
            logger.warning("EVAL_HF_PATH not set, cannot upload checkpoint for HaLLMark. Skipping.")
            hallmark_benchmarks = []

        # 7b. Launch all tasks in parallel:
        #   - S3 upload (starts immediately, async)
        #   - Gym eval (starts immediately, async)
        #   - HaLLMark (waits for S3 upload to finish, then calls API)
        gym_results = {}
        hallmark_results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Start S3 upload immediately (async)
            s3_future = None
            if hallmark_benchmarks and hf_path:
                s3_future = executor.submit(_upload_hf_checkpoint_to_s3, hf_path, int(step))

            # Start gym eval immediately (async, doesn't wait for S3)
            gym_future = None
            if eval_datasets:
                gym_future = executor.submit(
                    asyncio.run, _publish_and_collect(eval_datasets, nats_url, step)
                )

            # HaLLMark: wait for S3 upload to finish, then run
            hallmark_future = None
            if hallmark_benchmarks and s3_future:
                def _run_hallmark_after_upload():
                    # Block until S3 upload completes
                    s3_model_path = s3_future.result()
                    logger.info("S3 upload done (%s). Starting HaLLMark benchmarks...", s3_model_path)
                    return _run_hallmark_benchmarks(hallmark_benchmarks, int(step), checkpoint_s3_path=s3_model_path)
                hallmark_future = executor.submit(_run_hallmark_after_upload)

            # Collect results
            if gym_future:
                try:
                    gym_results = gym_future.result()
                except Exception as e:
                    logger.error("Gym eval failed: %s", e, exc_info=True)

            if hallmark_future:
                try:
                    hallmark_results = hallmark_future.result()
                    failed = [r for r in hallmark_results if "error" in r]
                    if failed:
                        logger.error(
                            "HaLLMark API calls failed for %d/%d group(s): %s",
                            len(failed), len(hallmark_results),
                            [r["error"] for r in failed],
                        )
                except Exception as e:
                    logger.error("HaLLMark eval failed: %s", e, exc_info=True)

            # Ensure S3 upload completes even if HaLLMark wasn't configured
            # (coordinator must not exit before upload finishes)
            if s3_future and not hallmark_future:
                try:
                    s3_future.result()
                except Exception as e:
                    logger.error("S3 upload failed: %s", e, exc_info=True)

        # 8. Persist metrics (gym + hallmark combined)
        try:
            _persist_eval_metrics(int(step), gym_results)
        except Exception as e:
            logger.error("gym metrics logging failed (non-fatal): %s", e, exc_info=True)


    finally:
        # Always clean up eval infra, whether we succeeded or crashed.
        try:
            _delete_eval_infra(k8s, namespace, job_name, step, eval_gyms)
        except Exception as e:
            logger.error("Failed to clean up eval infra: %s", e, exc_info=True)

    logger.info("Eval coordinator step=%s complete. Exiting.", step)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error("Eval coordinator failed: %s", e, exc_info=True)
        sys.exit(1)
