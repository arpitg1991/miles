"""Delta-rule kernel selection (CPU).

Ported from radixark/miles ``ff6193f26`` ``tests/fast/test_qwen_gdn_backend.py`` (PR #3634): the kernel
selection tests only. The fla against flashqla GPU tests stay in this fork's
``tests/fast/test_qwen_gdn_backend.py``.
"""

import os
import sys
import types

import pytest

from miles.kernels.attention.delta_rule import backend


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported GDN backend"):
        backend.get_chunk_gated_delta_rule("nope")


@pytest.mark.parametrize(
    "capability,user_value,expected", [((10, 0), None, "0"), ((9, 0), None, None), ((10, 0), "1", "1")]
)
def test_kda_uses_the_triton_backward_on_blackwell_unless_set(monkeypatch, capability, user_value, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "fla.ops.kda", types.SimpleNamespace(chunk_kda="chunk_kda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    if user_value is not None:
        monkeypatch.setenv("FLA_TILELANG", user_value)
    assert backend.get_chunk_kda.__wrapped__() == "chunk_kda"
    assert os.environ.get("FLA_TILELANG") == expected


def test_short_conv_backend_respects_fla_conv_backend(monkeypatch):
    monkeypatch.setenv("FLA_CONV_BACKEND", "cuda")
    assert backend.get_short_conv_backend.__wrapped__() == "cuda"
