"""GLM-5.3 --glm5-next-dsa-qp: the head-to-sequence layout round trip of the DSA core (CPU).

The query-parallel core moves the absorbed queries from the head split (each tensor-parallel rank
holds all ``S`` tokens of its ``64 / tp`` heads) to the sequence split (each rank holds its
sequence-parallel chunk of ``S / tp`` tokens on all 64 heads) with one ``all_to_all``, and moves the
attention output back with another. Both directions are a reshape, the collective, and a reshape.
This test simulates the equal-split ``all_to_all_single`` of the ``tp`` ranks in one process and
checks the two layouts against the direct slicing of the full ``[S, 64, D]`` tensor:

- rank ``r`` receives rows ``r * S / tp`` to ``(r + 1) * S / tp`` of every head, heads in global order;
- the inverse returns each rank's own head slice of all rows, bit for bit.

It also pins the flag, the spec hook, and the top-k adapter that lets ``--miles-dsa-topk-backend``
reach the kpool indexer. The GPU parity of the module itself is
``examples/arena/harbor-rl-glm53-flash/kdatp/dsa/t1_dsa_qp.py``.
"""

import argparse
import ast
from pathlib import Path

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.kernels.attention.dsa.topk import topk_with_scores, torch_dsa_topk
from miles.utils.arguments import get_miles_extra_args_provider
from miles_plugins.models.glm5_next.ops.qp_layout import (
    heads_to_sequence_a2a_input,
    heads_to_sequence_a2a_output,
    sequence_to_heads_a2a_input,
    sequence_to_heads_a2a_output,
)

register_cpu_ci(est_time=3, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[4]
TP, HEADS, DIM = 8, 64, 512


def _all_to_all(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """``all_to_all_single`` with equal splits on dim 0: output of rank r, chunk j = input of rank j, chunk r."""
    return [torch.stack([inputs[j][r] for j in range(TP)]) for r in range(TP)]


@pytest.mark.parametrize("seq_len", [TP, 4096, 4096 * 3])
def test_heads_to_sequence_gives_each_rank_its_sequence_chunk_on_all_heads(seq_len):
    full = torch.randn(seq_len, HEADS, DIM, dtype=torch.bfloat16)
    h_local, s_local = HEADS // TP, seq_len // TP
    per_rank = [full[:, r * h_local : (r + 1) * h_local].contiguous() for r in range(TP)]
    received = _all_to_all([heads_to_sequence_a2a_input(x, TP) for x in per_rank])
    for r in range(TP):
        got = heads_to_sequence_a2a_output(received[r])
        assert got.shape == (s_local, HEADS, DIM)
        assert torch.equal(got, full[r * s_local : (r + 1) * s_local])


@pytest.mark.parametrize("seq_len", [TP, 4096])
def test_sequence_to_heads_is_the_inverse(seq_len):
    full = torch.randn(seq_len, HEADS, DIM, dtype=torch.bfloat16)
    h_local, s_local = HEADS // TP, seq_len // TP
    chunks = [full[r * s_local : (r + 1) * s_local].contiguous() for r in range(TP)]
    received = _all_to_all([sequence_to_heads_a2a_input(y, TP) for y in chunks])
    for r in range(TP):
        got = sequence_to_heads_a2a_output(received[r])
        assert got.shape == (seq_len, h_local, DIM)
        assert torch.equal(got, full[:, r * h_local : (r + 1) * h_local])


def test_round_trip_is_identity_per_rank():
    seq_len = 4096
    h_local = HEADS // TP
    per_rank = [torch.randn(seq_len, h_local, DIM, dtype=torch.bfloat16) for _ in range(TP)]
    forward = [
        heads_to_sequence_a2a_output(y) for y in _all_to_all([heads_to_sequence_a2a_input(x, TP) for x in per_rank])
    ]
    back = [
        sequence_to_heads_a2a_output(x) for x in _all_to_all([sequence_to_heads_a2a_input(y, TP) for y in forward])
    ]
    for x, z in zip(per_rank, back, strict=True):
        assert torch.equal(x, z)


def test_rejects_a_sequence_that_the_ranks_cannot_split():
    with pytest.raises(AssertionError, match="not divisible"):
        heads_to_sequence_a2a_input(torch.zeros(TP + 1, HEADS // TP, DIM), TP)


def _parse(extra: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    return parser.parse_args([*extra, "--rollout-batch-size", "64"])


def test_flag_is_off_by_default():
    assert _parse([]).glm5_next_dsa_qp is False
    assert _parse(["--glm5-next-dsa-qp"]).glm5_next_dsa_qp is True


def test_spec_builder_passes_the_flag_and_the_module_takes_it():
    spec = (REPO_ROOT / "miles_plugins/models/glm5_next/glm5_next.py").read_text()
    assert '"query_parallel": bool(getattr(args, "glm5_next_dsa_qp", False))' in spec
    tree = ast.parse((REPO_ROOT / "miles_plugins/models/glm5_next/dsa.py").read_text())
    init = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "__init__"
        and any(a.arg == "query_parallel" for a in n.args.args)
    )
    assert init is not None


def test_both_cores_pass_the_forward_backend_and_the_top_k_reaches_the_indexer():
    tree = ast.parse((REPO_ROOT / "miles_plugins/models/glm5_next/dsa.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sparse_attention"
    ]
    assert len(calls) == 2, "one head-parallel and one query-parallel core"
    for call in calls:
        keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert keywords.get("forward_backend") == "self.sparse_attention_forward_backend"
    selects = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "kpool_select_topk"
    ]
    assert selects and all(
        {k.arg for k in call.keywords} >= {"token_ids", "topk_backend"} for call in selects
    ), "the kpool indexer gets the query ids and the top-k backend"


def test_topk_adapter_matches_torch_topk_and_masks_minus_one():
    logits = torch.randn(5, 16)
    logits[1, :] = float("-inf")
    logits[1, 3] = 2.0
    scores, indices = topk_with_scores(logits, 4, torch_dsa_topk)
    ref_scores, ref_indices = torch.topk(logits, 4, dim=-1)
    # torch_dsa_topk returns -1 for -inf slots; the adapter puts index 0 and -inf there, so the
    # expand kernel skips the slot exactly as the torch path does.
    finite = torch.isfinite(ref_scores)
    assert torch.equal(indices[finite], ref_indices[finite])
    assert torch.equal(scores[finite], ref_scores[finite])
    assert (indices[~finite] == 0).all() and torch.isneginf(scores[~finite]).all()
    assert indices.dtype == torch.int64
