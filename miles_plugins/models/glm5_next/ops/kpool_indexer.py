import torch

from miles.kernels.attention.dsa.kpool import append_tail_and_pad, pool_topk_to_token_fn, select_expand_tail
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn
from miles.utils.replay_base import indexer_replay_manager


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
    token_ids: torch.Tensor | None = None,
    topk_backend: str = "torch",
) -> torch.Tensor:
    """Top-k token indices of each query, ``[num_queries, 1, width]``.

    ``index_q`` and ``head_weights`` hold the queries; ``pooled_k``, ``cu_seqlens`` and ``pool_cu_seqlens``
    cover the whole packed sequence. ``token_ids`` gives the position of each query in the packed
    sequence (default: the queries are all tokens, in order; the query-parallel core passes its
    sequence-parallel chunk). ``topk_backend`` is ``--miles-dsa-topk-backend``: ``torch`` keeps
    ``torch.topk``; ``flashinfer`` selects the pools with ``flashinfer.top_k``.
    """
    num_tokens = index_q.shape[0]
    device = index_q.device
    if token_ids is None:
        token_ids = torch.arange(num_tokens, device=device)
    else:
        assert token_ids.shape == (num_tokens,), (token_ids.shape, num_tokens)
        assert not indexer_replay_manager.enabled, "the indexer replay holds the top-k of all tokens"
        token_ids = token_ids.to(device=device, dtype=torch.int64)
    pool_topk_fn = None if topk_backend == "torch" else get_dsa_topk_fn(topk_backend)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices].to(torch.int32)
    pool_base = pool_cu_seqlens[seq_indices].to(torch.int32)
    local_positions = (token_ids - seq_token_base).to(torch.int32)
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")

    if pooled_k.shape[0] > 0:
        # tilelang is GPU-only; importing it here keeps this module importable on CPU,
        # where tests/fast pins the per-sequence pool invariants.
        from miles.kernels.attention.dsa import indexer_logits

        with torch.no_grad():
            pool_logits = indexer_logits(
                index_q,
                pooled_k,
                head_weights,
                pool_base.to(torch.int32),
                (pool_base + eligible_pools).to(torch.int32),
            )
    else:
        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)

    if indexer_replay_manager.enabled:
        topk_fn = indexer_replay_manager.get_topk_fn(
            pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool, pool_topk_fn),
            return_probs=False,
        )
        tokens = topk_fn(pool_logits, index_topk)
        shortcut = (local_positions + 1) <= index_topk
        tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)
    else:
        tokens = select_expand_tail(
            pool_logits, seq_token_base, pool_base, local_positions, index_topk, kpool, pool_topk_fn
        )
    return tokens.unsqueeze(1)
