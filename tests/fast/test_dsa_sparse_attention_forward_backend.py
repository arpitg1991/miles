"""The DSA sparse-attention forward stays TileLang unless the run selects FlashMLA.

The kernel (PR #3608) picks FlashMLA by itself when ``flash_mla`` imports. The miles
images carry ``flash_mla``, so every model call site passes the
``--miles-dsa-sparse-attention-forward-backend`` value (default ``tilelang``) as
``forward_backend``, and the kernel default never applies.
"""

import argparse
import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.utils.arguments import get_miles_extra_args_provider

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAG = "--miles-dsa-sparse-attention-forward-backend"
CALL_SITES = (
    "miles_plugins/models/glm5/glm5.py",
    "miles_plugins/models/glm5_next/dsa.py",
    "miles_plugins/models/deepseek_v4/deepseek_v4.py",
)
SPEC_BUILDERS = (
    "miles_plugins/models/glm5/glm5.py",
    "miles_plugins/models/glm5_next/glm5_next.py",
    "miles_plugins/models/deepseek_v4/deepseek_v4.py",
)


def _parse(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser.parse_args([*extra, "--rollout-batch-size", "64"])


def test_flag_defaults_to_tilelang():
    assert _parse([]).miles_dsa_sparse_attention_forward_backend == "tilelang"
    assert _parse([FLAG, "flash_mla"]).miles_dsa_sparse_attention_forward_backend == "flash_mla"
    with pytest.raises(SystemExit):
        _parse([FLAG, "auto"])


@pytest.mark.parametrize("path", CALL_SITES)
def test_every_model_call_passes_the_forward_backend(path):
    tree = ast.parse((REPO_ROOT / path).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sparse_attention"
    ]
    assert calls
    for call in calls:
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        assert keywords.get("forward_backend") == "self.sparse_attention_forward_backend"


@pytest.mark.parametrize("path", SPEC_BUILDERS)
def test_spec_builders_read_the_flag(path):
    assert "args.miles_dsa_sparse_attention_forward_backend" in (REPO_ROOT / path).read_text()


def test_forward_backend_selects_the_kernel(monkeypatch):
    module = importlib.import_module("miles.kernels.attention.dsa.sparse_attention")
    calls = []

    def fake_flash_mla(q, kv, indices, sm_scale, d_v, attn_sink):
        calls.append(("flash_mla", tuple(q.shape), indices.shape[-1]))
        return q.new_zeros(*q.shape[:2], d_v), None, q.new_zeros(*q.shape[:2], dtype=torch.float32)

    def fake_tilelang(q, kv, indices, attn_sink, d_v, sm_scale):
        calls.append(("tilelang", tuple(q.shape), indices.shape[-1]))
        return q.new_zeros(*q.shape[:3], d_v), q.new_zeros(*q.shape[:3], dtype=torch.float32)

    tilelang_fwd = ModuleType("miles.kernels.attention.dsa.tilelang.sparse_attn_fwd")
    tilelang_fwd.sparse_attn_fwd_interface = fake_tilelang
    monkeypatch.setitem(sys.modules, tilelang_fwd.__name__, tilelang_fwd)
    monkeypatch.setattr(module, "flash_mla_sparse_fwd", fake_flash_mla)

    # GLM-5.3-Flash at TP8: 8 heads, 512 latent + 64 zero tail, top-k 2112 of the kpool indexer
    q = torch.zeros(1, 4, 8, 576, dtype=torch.bfloat16)
    kv = torch.zeros(1, 4, 1, 576, dtype=torch.bfloat16)
    indices = torch.full((1, 4, 1, 2112), -1, dtype=torch.int32)
    for backend in ("tilelang", "flash_mla"):
        out = module.sparse_attention(q, kv, indices, 576**-0.5, d_v=512, forward_backend=backend)
        assert out.shape == (1, 4, 8, 512)

    # TileLang pads top-k to 64; FlashMLA pads the heads to 64 and top-k to 128
    assert calls == [("tilelang", (1, 4, 8, 576), 2112), ("flash_mla", (4, 64, 576), 2176)]
