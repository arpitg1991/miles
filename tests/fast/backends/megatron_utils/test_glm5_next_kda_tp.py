"""GLM-5.3 --glm5-next-kda-tp: one packed-conv layout in every path (CPU).

The packed conv weight is ``[q; k; v]``, 8192 rows each. Under KDA tensor
parallelism rank r holds ``[q_r; k_r; v_r]``, head slice r of each part. Four
paths must agree on that layout, and each one fails silently when it does not:

- the DCP factory (checkpoint save and load),
- the weight-sync gather (SGLang weight update and HF export),
- the mbridge split and merge (HF to DCP conversion, bridge mode),
- the default module keeps the checkpoint names and shapes.

The GPU parity of the tensor-parallel module itself is T1 in
``examples/arena/harbor-rl-glm53-flash/kdatp/t1_parity.py``.
"""

from argparse import Namespace

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.dist_checkpointing.mapping import ShardedTensor  # noqa: E402

from miles.backends.megatron_utils.update_weight.common import (  # noqa: E402
    _check_and_fix_partition,
    _gather_with_stride,
)

CONV = "module.module.decoder.layers.0.self_attention.kda.conv1d.weight"
PARTS, TP, ROWS = 3, 8, 8192


def _full_conv() -> torch.Tensor:
    return torch.randn(PARTS * ROWS, 1, 4)


def _rank_shard(full: torch.Tensor, rank: int) -> torch.Tensor:
    return torch.cat([part.chunk(TP, dim=0)[rank] for part in full.chunk(PARTS, dim=0)], dim=0)


def test_weight_sync_gather_restores_the_packed_conv():
    full = _full_conv()
    shards = [_rank_shard(full, r) for r in range(TP)]
    stride, dim = _check_and_fix_partition(Namespace(swiglu=True), CONV, 3, 0)
    assert (stride, dim) == (3, 0)
    assert torch.equal(_gather_with_stride(shards, dim, stride), full)
    # A contiguous (stride 1) gather gives other weights with the same shape.
    assert not torch.equal(torch.cat(shards, dim=0), full)


def test_weight_sync_rejects_a_conv_without_stride_3():
    with pytest.raises(AssertionError, match="partition_stride=3"):
        _check_and_fix_partition(Namespace(swiglu=True), CONV, 1, 0)


@pytest.mark.parametrize("pp_offsets", [(), ((0, 5, 45),)])
def test_dcp_factory_saves_each_part_at_its_global_rows(pp_offsets):
    kda = pytest.importorskip("miles_plugins.models.glm5_next.kda")
    full = _full_conv()
    prepend = len(pp_offsets)
    covered = []
    for rank in range(TP):
        local = _rank_shard(full, rank)
        original = ShardedTensor.from_rank_offsets(
            "decoder.layers.0.self_attention.kda.conv1d.weight",
            local,
            *pp_offsets,
            (prepend, rank, TP),
            replica_id=(0, 0, 0),
            prepend_axis_num=prepend,
        )
        factory = kda.tp_strided_sharded_factory(original, pp_offsets, num_parts=PARTS)
        pieces = factory.build()
        assert len(pieces) == PARTS
        for piece in pieces:
            assert piece.key == original.key
            assert piece.global_shape[prepend:] == full.shape
            start = piece.global_offset[prepend]
            rows = piece.local_shape[0]
            assert torch.equal(piece.data, full[start : start + rows])
            covered.append((start, start + rows))
        assert torch.equal(factory.merge_fn([piece.data for piece in pieces]), local)
    covered.sort()
    assert covered[0][0] == 0 and covered[-1][1] == full.shape[0]
    assert all(end == nxt for (_, end), (nxt, _) in zip(covered, covered[1:], strict=False))


def test_mbridge_split_and_merge_use_the_same_layout():
    glm5_next = pytest.importorskip("miles_plugins.mbridge.glm5_next")
    bridge = object.__new__(glm5_next.Glm5NextBridge)
    name = "decoder.layers.0.self_attention.kda.conv1d.weight"
    full = _full_conv()
    local = torch.empty(PARTS * ROWS // TP, 1, 4)
    splits = bridge._weight_split_across_tp(name, full, local, TP)
    assert all(torch.equal(splits[r], _rank_shard(full, r)) for r in range(TP))
    assert torch.equal(bridge._weight_merge_across_tp(name, splits, local), full)
    # The default module keeps the full conv on each rank.
    replicated = bridge._weight_split_across_tp(name, full, torch.empty_like(full), TP)
    assert all(torch.equal(split, full) for split in replicated)


def test_default_module_keeps_the_checkpoint_layout():
    kda = pytest.importorskip("miles_plugins.models.glm5_next.kda")
    try:
        module = kda.Glm5NextKDA(
            hidden_size=4096, num_heads=64, head_dim=128, conv_kernel_size=4, gate_lower_bound=-5.0, rms_norm_eps=1e-5
        )
    except ImportError as exc:
        pytest.skip(str(exc))
    shapes = {name: (tuple(p.shape), p.dtype) for name, p in module.named_parameters()}
    # The 13 KDA tensors of the base, r43, r44 and r46 checkpoints (kdatp design, section 1).
    assert shapes == {
        "q_proj.weight": ((8192, 4096), torch.float32),
        "k_proj.weight": ((8192, 4096), torch.float32),
        "v_proj.weight": ((8192, 4096), torch.float32),
        "conv1d.weight": ((24576, 1, 4), torch.float32),
        "b_proj.weight": ((64, 4096), torch.float32),
        "f_a_proj.weight": ((128, 4096), torch.float32),
        "f_b_proj.weight": ((8192, 128), torch.float32),
        "g_a_proj.weight": ((128, 4096), torch.float32),
        "g_b_proj.weight": ((8192, 128), torch.float32),
        "A_log": ((64,), torch.float32),
        "dt_bias": ((8192,), torch.float32),
        "o_norm.weight": ((128,), torch.float32),
        "o_proj.weight": ((4096, 8192), torch.float32),
    }
    assert module.A_log.keep_in_fp32 and module.dt_bias.keep_in_fp32
    assert (module.local_num_heads, module.local_projection_size) == (64, 8192)
