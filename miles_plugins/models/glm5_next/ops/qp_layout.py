"""Layout reshapes of the query-parallel DSA core (``--glm5-next-dsa-qp``).

The head-parallel layout holds all ``S`` tokens of this rank's ``h_local`` heads: ``[S, h_local, D]``.
The query-parallel layout holds this rank's sequence-parallel chunk of ``S / tp`` tokens on all
``tp * h_local`` heads: ``[S / tp, tp * h_local, D]``. One equal-split ``all_to_all`` over the
tensor-parallel group moves between them; these four functions are the reshapes around it. They
use torch only, so the CPU tests can pin the layout without Megatron.
"""

import torch


def heads_to_sequence_a2a_input(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    """[S, h_local, D] in the head-parallel layout -> the all-to-all input [tp, S / tp, h_local, D].

    Chunk ``c`` of the sequence (rows ``c * S / tp`` to ``(c + 1) * S / tp``, the sequence-parallel
    chunk of rank ``c``) goes to rank ``c``. The view is free: the chunks are contiguous rows.
    """
    seq_len, h_local, dim = x.shape
    assert seq_len % tp_size == 0, f"sequence length {seq_len} is not divisible by tp={tp_size}"
    return x.view(tp_size, seq_len // tp_size, h_local, dim)


def heads_to_sequence_a2a_output(y: torch.Tensor) -> torch.Tensor:
    """The all-to-all output [tp, S / tp, h_local, D] (dim 0 = source rank = head group) -> [S / tp, tp * h_local, D].

    Head ``j * h_local + i`` is head ``i`` of rank ``j``, the global head order of the column-parallel
    projections.
    """
    tp_size, seq_local, h_local, dim = y.shape
    return y.permute(1, 0, 2, 3).reshape(seq_local, tp_size * h_local, dim)


def sequence_to_heads_a2a_input(y: torch.Tensor, tp_size: int) -> torch.Tensor:
    """[S / tp, tp * h_local, D] -> the all-to-all input [tp, S / tp, h_local, D]: head group ``j`` goes to rank ``j``."""
    seq_local, heads, dim = y.shape
    assert heads % tp_size == 0, f"{heads} heads are not divisible by tp={tp_size}"
    return y.view(seq_local, tp_size, heads // tp_size, dim).permute(1, 0, 2, 3).contiguous()


def sequence_to_heads_a2a_output(x: torch.Tensor) -> torch.Tensor:
    """The all-to-all output [tp, S / tp, h_local, D] (dim 0 = source rank = sequence chunk) -> [S, h_local, D]."""
    tp_size, seq_local, h_local, dim = x.shape
    return x.view(tp_size * seq_local, h_local, dim)
