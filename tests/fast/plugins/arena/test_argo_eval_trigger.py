"""Unit tests for the fire-and-forget per-checkpoint eval trigger.

The module is import-light (no miles/kubernetes at import time), so these run
without the GPU runtime. The core guarantee: a submission failure (or any bad
input) never propagates out of ``submit_eval`` -- training must not be
interrupted.

Run: python -m pytest tests/fast/plugins/arena/test_argo_eval_trigger.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miles_plugins.arena.nats_arena import argo_eval_trigger as t


def _args(**kw):
    base = dict(
        wandb_project="arena-slime-deployer",
        wandb_group="sbofan-run-xyz",
        wandb_random_suffix=False,
        save_hf="/mnt/scratch/hf/rollout_{rollout_id}",
        user="sbofan",
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def _eval_env(monkeypatch):
    monkeypatch.setenv(t._ENV_TASKS, "amzn_arena_tasks/chakra_productivity_alder_ridge")
    monkeypatch.setenv("USER", "sbofan")
    monkeypatch.setenv("JOBNAME", "sbofan-run-xyz")


def test_noop_when_not_enabled(monkeypatch):
    # No ARENA_EVAL_TASKS -> opt-in off -> nothing resolved or submitted.
    monkeypatch.delenv(t._ENV_TASKS, raising=False)
    resolve = MagicMock()
    submit = MagicMock()
    monkeypatch.setattr(t, "_resolve_hf_checkpoint", resolve)
    monkeypatch.setattr(t, "_submit", submit)

    t.submit_eval(_args(), 3)

    resolve.assert_not_called()
    submit.assert_not_called()


def test_skips_when_no_checkpoint(monkeypatch, _eval_env):
    monkeypatch.setattr(t, "_resolve_hf_checkpoint", lambda a, r: None)
    submit = MagicMock()
    monkeypatch.setattr(t, "_submit", submit)

    t.submit_eval(_args(), 3)

    submit.assert_not_called()


def test_submits_with_wandb_merge_coords(monkeypatch, _eval_env):
    monkeypatch.setattr(t, "_resolve_hf_checkpoint", lambda a, r: "/mnt/hf/rollout_3")
    captured = {}
    monkeypatch.setattr(t, "_submit", lambda wf: captured.update(wf) or "wf-1")

    t.submit_eval(_args(), 3)

    params = {p["name"]: p["value"] for p in captured["spec"]["arguments"]["parameters"]}
    # W&B coords point the hook at the TRAINING run at this step.
    assert params["experiment"] == "arena-slime-deployer"
    assert params["run_name"] == "sbofan-run-xyz"
    assert params["step"] == "3"
    assert params["checkpoint_path"] == "/mnt/hf/rollout_3"
    # Piled-up evals must never preempt the trainer.
    assert params["priority_class_name"] == "preemptible"
    # Kyverno ownership label + template ref.
    assert captured["metadata"]["labels"][t._SUBMITTER_LABEL] == "sbofan"
    assert captured["spec"]["workflowTemplateRef"]["name"] == t._EVAL_TEMPLATE


def test_random_suffix_disables_wandb_merge_but_still_evals(monkeypatch, _eval_env):
    # A non-deterministic trainer run name cannot be matched, so we drop the
    # W&B coords (empty -> hook skips) rather than pollute a wrong run.
    monkeypatch.setattr(t, "_resolve_hf_checkpoint", lambda a, r: "/mnt/hf/rollout_3")
    captured = {}
    monkeypatch.setattr(t, "_submit", lambda wf: captured.update(wf) or "wf-1")

    t.submit_eval(_args(wandb_random_suffix=True), 3)

    params = {p["name"]: p["value"] for p in captured["spec"]["arguments"]["parameters"]}
    assert "experiment" not in params
    assert "run_name" not in params
    assert params["checkpoint_path"] == "/mnt/hf/rollout_3"  # eval still runs


def test_missing_random_suffix_attr_skips_wandb_merge(monkeypatch, _eval_env):
    # Defense-in-depth: if the args object lacks wandb_random_suffix entirely,
    # assume the run name is non-deterministic (the trainer argparse default is
    # True) and DROP the W&B coords rather than merge into a possibly-wrong run.
    monkeypatch.setattr(t, "_resolve_hf_checkpoint", lambda a, r: "/mnt/hf/rollout_3")
    captured = {}
    monkeypatch.setattr(t, "_submit", lambda wf: captured.update(wf) or "wf-1")

    args = _args()
    del args.wandb_random_suffix  # attribute absent

    t.submit_eval(args, 3)

    params = {p["name"]: p["value"] for p in captured["spec"]["arguments"]["parameters"]}
    assert "experiment" not in params
    assert "run_name" not in params
    assert params["checkpoint_path"] == "/mnt/hf/rollout_3"  # eval still runs


def test_never_raises_when_submit_fails(monkeypatch, _eval_env):
    # THE guarantee: a submit failure degrades to a skip, never breaks training.
    monkeypatch.setattr(t, "_resolve_hf_checkpoint", lambda a, r: "/mnt/hf/rollout_3")

    def _boom(_wf):
        raise RuntimeError("k8s API unreachable")

    monkeypatch.setattr(t, "_submit", _boom)

    t.submit_eval(_args(), 3)  # must not raise
