from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.cp_utils import slice_with_cp
from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

    def record(self, value):
        self.recorded.append(value)


def test_register_replay_list_sequential_falls_back_to_enum_when_no_stream_idx():
    replay_data = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    replays = [_Replay(), _Replay(), _Replay()]

    register_replay_list_sequential(replays, replay_data)

    for replay_idx, replay in enumerate(replays):
        assert len(replay.recorded) == 1
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay_idx])


def test_register_replay_list_sequential_uses_stream_idx_when_set():
    # PP case: 2 local modules at global streams 1 and 3 out of 4.
    replay_data = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    replays = [_Replay(stream_idx=1), _Replay(stream_idx=3)]

    register_replay_list_sequential(replays, replay_data)

    torch.testing.assert_close(replays[0].recorded[0], replay_data[:, 1])
    torch.testing.assert_close(replays[1].recorded[0], replay_data[:, 3])


def test_register_replay_list_sequential_rejects_out_of_range_stream_idx():
    replay_data = torch.zeros(5, 4, 2)
    replays = [_Replay(stream_idx=4)]

    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(replays, replay_data)


# ----------------------------- fill_replay_data: thd row walk against the old fill -----------------------------

NUM_LAYERS = 45
TOPK = 8
NUM_EXPERTS = 288
# GLM-5.3-Flash with PP 4 (r47): layers 0-2 are dense, so the stages hold 8, 11, 11 and 12 local MoE layers.
STAGE_MOE_LAYERS = [range(3, 11), range(11, 22), range(22, 33), range(33, 45)]


def _reference_fill_replay_data(
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    data_key,
    replay_list,
    register_replay_list_func,
    if_sp_region=True,
    indices_are_token_positions=False,
):
    """fill_replay_data at 4716a367a (pad, concatenate, pad, then slice), kept as the reference.

    Only the thd branch is here; the bshd branch did not change.
    """
    for iterator in data_iterator:
        iterator.reset()

    parallel_state = parallel.get_parallel_state()
    tp_rank = parallel_state.tp.rank
    tp_size = parallel_state.tp.size
    qkv_format = args.qkv_format

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
        replay_data = [pad_func(r, 1) for r in replay_data]

        cp_size = parallel_state.cp.size
        cp_rank = parallel_state.cp.rank
        pad_size = parallel_state.tp.size * args.data_pad_size_multiplier
        if args.allgather_cp and cp_size > 1:
            replay_data = torch.cat(replay_data, dim=0)
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - replay_data.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                replay_data = pad_func(replay_data, pad)
            replay_data = replay_data.chunk(cp_size, dim=0)[cp_rank]
        else:
            replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
            if indices_are_token_positions:
                # map indices to thd format
                offset = 0
                for i, r in enumerate(replay_data):
                    replay_data[i] = torch.where(r != -1, r + offset, r)
                    offset += r.shape[0]
            replay_data = torch.cat(replay_data, dim=0)
            pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
            if pad != 0:
                replay_data = pad_func(replay_data, pad)

        if getattr(args, "sequence_parallel", False) and if_sp_region:
            seqlen = replay_data.size(0)
            assert seqlen % tp_size == 0
            start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
            replay_data = replay_data[start:end]

        register_replay_list_func(replay_list, replay_data, models=models)

    del rollout_data[data_key]

    for iterator in data_iterator:
        iterator.reset()


class _MicroBatchIterator:
    def __init__(self, micro_batches):
        self.micro_batches = micro_batches
        self.index = 0

    def reset(self):
        self.index = 0

    def get_next(self, keys):
        batch = self.micro_batches[self.index]
        self.index += 1
        return {key: batch[key] for key in keys}


def _micro_batches(lengths_per_batch, *, dtype, token_positions, seed):
    """Random routing rows with -1 rows and -1 slots; token positions for the indexer case."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for lengths in lengths_per_batch:
        routing = []
        for n in lengths:
            high = max(n, 1) if token_positions else NUM_EXPERTS
            r = torch.randint(0, high, (n, NUM_LAYERS, TOPK), generator=generator, dtype=torch.int64)
            r[torch.rand((n, NUM_LAYERS, TOPK), generator=generator) < 0.05] = -1
            r[torch.rand((n,), generator=generator) < 0.1] = -1
            routing.append(r.to(dtype))
        batches.append({"data": routing, "tokens": [torch.zeros(n + 1) for n in lengths], "max_seq_lens": None})
    return batches


def _run_fill(fill, *, args, micro_batches, stage_layers, if_sp_region, token_positions):
    replays = [_Replay(stream_idx=layer) for layer in stage_layers]
    fill(
        args=args,
        models=None,
        data_iterator=[_MicroBatchIterator(micro_batches)],
        num_microbatches=[len(micro_batches)],
        rollout_data={"data": None},
        data_key="data",
        replay_list=replays,
        register_replay_list_func=register_replay_list_sequential,
        if_sp_region=if_sp_region,
        indices_are_token_positions=token_positions,
    )
    return [recorded for replay in replays for recorded in replay.recorded]


# "exact": the packed rows are a multiple of every pad size here, so the tail pad is 0. "ragged": a 0-row sample,
# a 1-row sample and lengths that need CP and tail pad rows.
LENGTHS = {"exact": [[7, 15], [31]], "ragged": [[0, 1, 6, 33], [100, 2]]}


@pytest.mark.parametrize("mode", ["routing-int16", "routing-int32", "indexer"])
@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("sequence_parallel", [True, False])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("lengths", sorted(LENGTHS))
@pytest.mark.parametrize("pad_multiplier", [1, 4])
def test_thd_fill_matches_the_old_fill(
    monkeypatch, mode, tp_size, sequence_parallel, cp_size, allgather_cp, lengths, pad_multiplier
):
    """Every recorded replay tensor is byte-identical to the one of the pad-concat-slice fill."""
    token_positions = mode == "indexer"
    dtype = torch.int16 if mode == "routing-int16" else torch.int32
    # The indexer manager replays outside the SP region; its streams are layers too.
    if_sp_region = not token_positions
    micro_batches = _micro_batches(
        LENGTHS[lengths], dtype=dtype, token_positions=token_positions, seed=tp_size * 100 + cp_size
    )
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=allgather_cp,
        data_pad_size_multiplier=pad_multiplier,
        sequence_parallel=sequence_parallel,
    )
    stages = [range(NUM_LAYERS)] if token_positions else STAGE_MOE_LAYERS
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            state = SimpleNamespace(
                tp=SimpleNamespace(rank=tp_rank, size=tp_size), cp=SimpleNamespace(rank=cp_rank, size=cp_size)
            )
            monkeypatch.setattr(parallel, "_parallel_state", state)
            for stage_layers in stages:
                kwargs = dict(
                    args=args,
                    micro_batches=micro_batches,
                    stage_layers=stage_layers,
                    if_sp_region=if_sp_region,
                    token_positions=token_positions,
                )
                new = _run_fill(fill_replay_data, **kwargs)
                old = _run_fill(_reference_fill_replay_data, **kwargs)
                assert len(new) == len(old) == len(stage_layers) * len(micro_batches)
                for a, b in zip(new, old, strict=True):
                    assert a.dtype == b.dtype == dtype
                    assert torch.equal(a, b), (cp_rank, tp_rank, stage_layers)
