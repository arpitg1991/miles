"""GLM-5.3 --glm5-next-dsa-qp: the head-to-sequence layout round trip of the DSA core (CPU).

The query-parallel core moves the absorbed queries from the head split (each tensor-parallel rank
holds all ``S`` tokens of its ``64 / tp`` heads) to the sequence split (each rank holds its
sequence-parallel chunk of ``S / tp`` tokens on all 64 heads) with one ``all_to_all``, and moves the
attention output back with another. Both directions are a reshape, the collective, and a reshape.
This test simulates the equal-split ``all_to_all_single`` of the ``tp`` ranks in one process and
checks the two layouts against the direct slicing of the full ``[S, 64, D]`` tensor:

- rank ``r`` receives rows ``r * S / tp`` to ``(r + 1) * S / tp`` of every head, heads in global order;
- the inverse returns each rank's own head slice of all rows, bit for bit.

The GPU parity of the module itself is in ``examples/arena/harbor-rl-glm53-flash/kdatp/dsa/``.
"""

import pytest
import torch

pytest.importorskip("megatron.core")

from miles_plugins.models.glm5_next.dsa import (  # noqa: E402
    heads_to_sequence_a2a_input,
    heads_to_sequence_a2a_output,
    sequence_to_heads_a2a_input,
    sequence_to_heads_a2a_output,
)

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
