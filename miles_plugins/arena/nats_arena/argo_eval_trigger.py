"""Fire-and-forget per-checkpoint eval trigger (miles -> Argo hosted-eval).

Called from ``miles_plugins.arena.train_async_arena``'s save block, immediately
after ``save_model(rollout_id)`` (which blocks until both the DCP
``iter_N`` and HF ``hf/rollout_N`` checkpoints are flushed). This function:

  1. Resolves the HF checkpoint for ``rollout_id`` (from ``--save-hf``).
  2. Submits an ``arena-hosted-eval`` Argo Workflow that serves the checkpoint
     and runs a held-out eval set.
  3. Returns immediately. The Workflow's ``arena_wandb_metrics`` hook publishes
     ``validation/*`` INTO the training W&B run.

Triggering from the save block (rather than miles' ``--eval-function-path`` /
``--eval-interval`` seam) is deliberate: that seam makes the trainer's
argument validation require ``eval_datasets``, which we do not use — our eval
runs out-of-band in Argo, not through the trainer's rollout engine.

W&B same-run merge: the metrics hook resolves the run by *display name* and
resumes it. We pass ``experiment`` = the trainer's ``wandb_project`` and
``run_name`` = ``wandb_group`` (the run's display name when
``--disable-wandb-random-suffix`` is set) + ``step`` = ``rollout_id``, so the
eval lands in the trainer's own run on the ``validation/step`` axis. The trainer
inits W&B in ``mode="shared", x_primary=True``, so a second writer is supported.

Opt-in via ``ARENA_EVAL_TASKS``. Never raises: every failure degrades to a skip
so training is never interrupted.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["submit_eval"]

# Argo Workflow CR coordinates.
_ARGO_GROUP = "argoproj.io"
_ARGO_VERSION = "v1alpha1"
_ARGO_PLURAL = "workflows"
# WorkflowTemplate to instantiate (AREnATasksApps arena-hosted-eval chart).
_EVAL_TEMPLATE = "arena-hosted-eval"
_NAMESPACE = os.environ.get("NAMESPACE", "arena-tasks")
# Kyverno stamp-submitter / workload-owner policies confine each submitted
# workflow to its submitter; a missing label is rejected.
_SUBMITTER_LABEL = "arena.agif.amazon.dev/submitter"

# Eval-shape knobs (env-configurable; the trainer pod exports these). Only the
# run-specific choices live here — everything W&B/checkpoint is derived from
# ``args`` so it cannot drift from the training run. ARENA_EVAL_TASKS also acts
# as the opt-in switch: unset -> no eval is triggered.
_ENV_TASKS = "ARENA_EVAL_TASKS"                  # required: held-out eval task ref(s)
_ENV_MODEL_PROFILE = "ARENA_EVAL_MODEL_PROFILE"  # hosted-eval serving profile
_ENV_ENGINE = "ARENA_EVAL_ENGINE"                # vllm | sglang
_ENV_EVAL_ARGS = "ARENA_EVAL_ARGS"               # bounded Inspect args
_ENV_PRIORITY = "ARENA_EVAL_PRIORITY"            # normal | high | preemptible

# hosted-eval currently ships a single serving profile; default to it.
_DEFAULT_MODEL_PROFILE = "qwen3-5-b200-tp2"
# sglang matches the training rollout serving (tool-call / reasoning parsers).
_DEFAULT_ENGINE = "sglang"
# Bounded by default so one eval stays a small fraction of the save cadence.
_DEFAULT_EVAL_ARGS = "--limit 32"
# Preemptible so piled-up evals can never starve or preempt the trainer.
_DEFAULT_PRIORITY = "preemptible"


def _resolve_hf_checkpoint(args: Any, rollout_id: int) -> str | None:
    """Return the HF checkpoint dir for ``rollout_id``, or None if absent."""
    from miles_plugins.arena.nats_arena.eval_rollout import _find_latest_hf_checkpoint

    return _find_latest_hf_checkpoint(args, rollout_id)


def _wandb_coords(args: Any, rollout_id: int) -> dict[str, str]:
    """Derive the three W&B params that merge the eval into the training run.

    Returns empty ``experiment``/``run_name`` (which makes the hook skip W&B)
    when the run name is not deterministic, so we never pollute a wrong run.
    """
    project = getattr(args, "wandb_project", None) or ""
    # Default True (the trainer argparse default for this dest): if the attribute
    # is ever absent, assume the run name is non-deterministic and SKIP the
    # merge — the safe direction. Defaulting False would proceed to merge and
    # risk polluting a wrong run, exactly what this guard exists to prevent.
    if getattr(args, "wandb_random_suffix", True):
        logger.warning(
            "eval-trigger: wandb_random_suffix is on; the trainer run name is "
            "non-deterministic, so eval metrics cannot merge by name. Pass "
            "--disable-wandb-random-suffix to enable the merge. Skipping W&B "
            "coords for this eval."
        )
        return {"experiment": "", "run_name": "", "step": str(rollout_id)}
    run_name = getattr(args, "wandb_group", None) or ""
    # NOTE: the W&B entity/team is intentionally not forwarded here. The eval's
    # metrics hook resolves the run under WANDB_ENTITY (default "arena"), which
    # must match the trainer's entity for the merge to land in the same run.
    # Today both default to "arena" (trainer: --wandb-team default None ->
    # account default entity; hook: DEFAULT_WANDB_ENTITY="arena"), so they
    # align. A non-default --wandb-team would move the trainer run and orphan
    # the eval metrics into "arena"; forwarding the team to the eval pod's
    # WANDB_ENTITY is a hosted-eval (AREnATasksApps) follow-up, not doable from
    # this direct-submit path (arena-hosted-eval declares no entity param).
    return {"experiment": project, "run_name": run_name, "step": str(rollout_id)}


def _build_workflow(args: Any, rollout_id: int, hf_path: str) -> dict[str, Any]:
    """Build a thin Argo Workflow CR that instantiates arena-hosted-eval."""
    username = os.environ.get("USER") or getattr(args, "user", None) or "unknown"

    params: dict[str, str] = {
        "checkpoint_path": hf_path,
        "model_profile": os.environ.get(_ENV_MODEL_PROFILE, _DEFAULT_MODEL_PROFILE),
        "inference_engine": os.environ.get(_ENV_ENGINE, _DEFAULT_ENGINE),
        "priority_class_name": os.environ.get(_ENV_PRIORITY, _DEFAULT_PRIORITY),
        "tasks": os.environ.get(_ENV_TASKS, ""),
        "eval_args": os.environ.get(_ENV_EVAL_ARGS, _DEFAULT_EVAL_ARGS),
        "username": username,
        **_wandb_coords(args, rollout_id),
    }
    param_list = [{"name": k, "value": v} for k, v in params.items() if v != ""]

    job = os.environ.get("JOBNAME") or getattr(args, "experiment_name", "") or "slime"
    return {
        "apiVersion": f"{_ARGO_GROUP}/{_ARGO_VERSION}",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"{job}-eval-s{rollout_id}-",
            "namespace": _NAMESPACE,
            "labels": {_SUBMITTER_LABEL: username, "arena.job": job},
        },
        "spec": {
            "workflowTemplateRef": {"name": _EVAL_TEMPLATE},
            "arguments": {"parameters": param_list},
        },
    }


def _submit(workflow: dict[str, Any]) -> str:
    """Create the Workflow CR via the in-cluster SA (SDK, no argo/kubectl CLI)."""
    import kubernetes

    # Reload each call: projected SA tokens rotate (~1h); a cached client sends
    # a stale token and 401s on later triggers in a long run.
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException:
        kubernetes.config.load_kube_config()

    api = kubernetes.client.CustomObjectsApi()
    created = api.create_namespaced_custom_object(
        group=_ARGO_GROUP,
        version=_ARGO_VERSION,
        plural=_ARGO_PLURAL,
        namespace=workflow["metadata"]["namespace"],
        body=workflow,
    )
    return created["metadata"]["name"]


def submit_eval(args: Any, rollout_id: int) -> None:
    """Fire-and-forget: submit one hosted-eval workflow for this checkpoint.

    Opt-in: no-op unless ``ARENA_EVAL_TASKS`` is set. Call from the trainer's
    save block, after the (blocking) ``save_model`` for ``rollout_id``.
    """
    if not os.environ.get(_ENV_TASKS):
        return  # eval-on-checkpoint not enabled for this run

    # Everything below is best-effort: a failure here MUST NOT break training.
    try:
        hf_path = _resolve_hf_checkpoint(args, rollout_id)
        if not hf_path:
            logger.warning(
                "eval-trigger: no HF checkpoint for rollout_id=%d; skipping.",
                rollout_id,
            )
            return

        name = _submit(_build_workflow(args, rollout_id, hf_path))
        logger.info(
            "eval-trigger: submitted %s for rollout_id=%d (checkpoint=%s)",
            name,
            rollout_id,
            hf_path,
        )
    except Exception:  # noqa: BLE001 - fire-and-forget: never interrupt training
        logger.warning(
            "eval-trigger: submit failed for rollout_id=%d; skipping.",
            rollout_id,
            exc_info=True,
        )
