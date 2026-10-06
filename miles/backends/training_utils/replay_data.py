from collections.abc import Callable
from typing import Protocol

import torch

from .cp_utils import slice_with_cp
from .parallel import get_parallel_state


class RegisterReplayListFunc(Protocol):
    def __call__(self, replay_list: list, replay_data: torch.Tensor, **kwargs) -> None: ...


def register_replay_list_sequential(replay_list, replay_data, **_kwargs):
    """Map replay streams to registered modules.

    Each replay records `replay_data[:, replay.stream_idx]` if `stream_idx` is
    set (used for sparse layer layouts under PP/VPP, where the global replay
    tensor contains more streams than this rank registered). Otherwise falls
    back to 1:1 enumeration order.
    """
    for replay_idx, replay in enumerate(replay_list):
        stream_idx = replay.stream_idx if replay.stream_idx is not None else replay_idx
        if not 0 <= stream_idx < replay_data.shape[1]:
            raise AssertionError(
                f"replay stream_idx {stream_idx} out of range " f"(replay_data has {replay_data.shape[1]} streams)"
            )
        replay.record(replay_data[:, stream_idx])


def fill_replay_data(
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    data_key: str,
    replay_list: list,
    register_replay_list_func: RegisterReplayListFunc,
    if_sp_region=True,
    indices_are_token_positions=False,
):
    """Load rollout replay tensors into module replay queues.

    `rollout_data[data_key]` contains one tensor per sample with shape
    `[num_tokens - 1, num_streams, topk]`. This function replays the training
    data iterator to process those tensors in the same microbatch order as
    log-prob and train forwards, pads/slices them to match the local CP/SP
    token layout, and then delegates stream-to-module mapping to
    `register_replay_list_func`.
    """
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")

    for iterator in data_iterator:
        iterator.reset()

    parallel_state = get_parallel_state()
    tp_rank = parallel_state.tp.rank
    tp_size = parallel_state.tp.size
    qkv_format = args.qkv_format
    # sequence_parallel is Megatron-only; FSDP has tp_size == 1 so the slice is a no-op there.
    sequence_parallel = getattr(args, "sequence_parallel", False) and if_sp_region

    def sp_rows(seqlen: int) -> slice:
        if not sequence_parallel:
            return slice(0, seqlen)
        assert seqlen % tp_size == 0
        return slice(seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1))

    def pad_func(data, pad):
        _, num_layers, topk = data.shape
        pad_tensor = torch.full(
            (pad, num_layers, topk),
            fill_value=-1,
            device=data.device,
            dtype=data.dtype,
        )
        return torch.cat([data, pad_tensor], dim=0)

    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
        replay_data = batch[data_key]
        tokens = batch["tokens"]
        assert len(replay_data) == len(tokens)
        for a, b in zip(replay_data, tokens, strict=False):
            assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

        # TODO: maybe extract a common process function for here and get_batch?
        cp_size = parallel_state.cp.size
        cp_rank = parallel_state.cp.rank
        if qkv_format == "bshd":
            # Pad replay data to align with the token batch's final token. The padded token is masked from loss.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            max_seqlen = batch["max_seq_lens"][0]
            if args.allgather_cp and cp_size > 1:
                assert max_seqlen % cp_size == 0, f"max_seqlen {max_seqlen} must be divisible by cp_size {cp_size}"
                local_len = max_seqlen // cp_size
                start = cp_rank * local_len
                replay_data = [pad_func(r, max_seqlen - r.size(0))[start : start + local_len] for r in replay_data]
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
            replay_data = torch.stack(replay_data, dim=0)
            batch_size, seqlen, num_layers, topk = replay_data.shape
            replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
            replay_data = replay_data[sp_rows(replay_data.size(0))]
        else:
            replay_data = _thd_local_rows(
                replay_data,
                pad_size=parallel_state.tp.size * args.data_pad_size_multiplier,
                cp_size=cp_size,
                cp_rank=cp_rank,
                allgather_cp=args.allgather_cp,
                sp_rows=sp_rows,
                indices_are_token_positions=indices_are_token_positions,
            )

        register_replay_list_func(replay_list, replay_data, models=models)

    del rollout_data[data_key]

    for iterator in data_iterator:
        iterator.reset()


def _thd_local_rows(
    replay_data: list[torch.Tensor],
    *,
    pad_size: int,
    cp_size: int,
    cp_rank: int,
    allgather_cp: bool,
    sp_rows: Callable[[int], slice],
    indices_are_token_positions: bool,
) -> torch.Tensor:
    """This rank's rows of the packed ``thd`` replay sequence, copied once.

    The packed sequence is each sample plus one -1 row (the loss-masked last
    token), in the local CP layout, then -1 rows up to the pad multiple.
    ``sp_rows`` keeps this rank's sequence-parallel part of the local rows.
    Building the whole sequence first copies every row three times, and a
    rank keeps only about 1/tp of the rows, so this walk copies only the kept
    rows.

    Port of the row-range fill in slime
    ``prepare_routed_experts_for_routing_replay``
    (``slime/backends/megatron_utils/cp_utils.py`` at 8088a4b, from THUDM/slime
    PRs #2175 and #2410, Apache-2.0). Changes: -1 pad rows, not slime's expert
    pattern, so that the replay tensors stay the same; and the token-position
    offset of the indexer replay.
    """
    allgather = allgather_cp and cp_size > 1
    # Per sample: (routing, chunks). A chunk is a row range of the sample plus its -1 row, in local order.
    # Zigzag CP pads each sample with -1 rows to 2 * cp_size chunks; rows at or past len(routing) are -1.
    samples = []
    for routing in replay_data:
        token_len = routing.size(0) + 1
        if allgather or cp_size == 1:
            chunks = [(0, token_len)]
        else:
            chunk = (token_len + 2 * cp_size - 1) // (2 * cp_size)
            chunks = [
                (chunk * cp_rank, chunk * (cp_rank + 1)),
                (chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)),
            ]
        samples.append((routing, chunks))
    total = sum(end - start for _, chunks in samples for start, end in chunks)
    multiple = pad_size * cp_size if allgather else pad_size
    padded = total + (multiple - total % multiple) % multiple
    # All-gather CP pads the whole packed sequence, then this CP rank keeps one of cp_size equal parts.
    local = padded // cp_size if allgather else padded
    base = local * cp_rank if allgather else 0
    rows = sp_rows(local)
    start, end = base + rows.start, base + rows.stop

    out = replay_data[0].new_empty((end - start, *replay_data[0].shape[1:]))
    pos = 0  # row of the packed sequence where the current chunk starts
    for routing, chunks in samples:
        # The indexer replays token positions: rebase them onto the sample start in the packed sequence. The
        # all-gather CP path has never rebased them, so it does not here, and its replay tensors stay the same.
        offset = pos if indices_are_token_positions and not allgather else None
        for chunk_start, chunk_end in chunks:
            lo, hi = max(pos, start), min(pos + chunk_end - chunk_start, end)
            if lo < hi:
                src = chunk_start + lo - pos
                real = max(0, min(hi - lo, routing.size(0) - src))
                piece = routing[src : src + real]
                if offset is not None:
                    piece = torch.where(piece != -1, piece + offset, piece)
                out[lo - start : lo - start + real] = piece
                out[lo - start + real : hi - start] = -1
            pos += chunk_end - chunk_start
    out[max(total, start) - start :] = -1
    return out
