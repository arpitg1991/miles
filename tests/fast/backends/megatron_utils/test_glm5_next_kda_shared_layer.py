"""GLM-5.3 KDA on the shared head-sharded layer: checkpoint layout, weight sync and mbridge (CPU, gloo).

The GLM-5.3 checkpoints (the base DCP, r43/r45 ``iter_0000039``, the r44 and r46 saves) hold each KDA
tensor of the old replicated layer as one chunk under ``self_attention.kda.``, with one packed
``conv1d.weight`` ``[q; k; v]``. The shared layer names its parameters ``self_attention.linear_attn.*``
and each rank holds one head slice. Each path below fails silently when its layout is wrong:

- load: an old-layout save loads into the TP-sharded layer, each shard bit for bit;
- save: a save of the layer has the old keys, global shapes and dtypes. It loads back into the layer,
  and into the old layout (rollback), bit for bit;
- weight sync: the production gather plus ``convert_glm5_next_to_hf`` gives the HF tensors of the old
  layout, bit for bit;
- mbridge: the two converters give the same HF names, and the TP split and merge restore each tensor;
- offline conversion: ``tools/convert_torch_dist_to_hf.py`` and ``tools/convert_torch_dist_to_hf_ray.py``
  read the stored keys of a DCP and write the HF tensors of the weight sync, bit for bit.

The GPU parity of the layer is ``tests/fast-gpu/test_delta_rule_head_sharded.py``.
"""

import contextlib
import pickle
import socket
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
pytest.importorskip("fla")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

PREFIX = "decoder.layers.3.self_attention."
WORLD = 4
SMALL = dict(hidden=64, heads=8, head_dim=16)
# The real GLM-5.3-Flash KDA sizes of the checkpoints.
REAL = dict(hidden=4096, heads=64, head_dim=128)
SHARDED_ROWS = ("q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj", "g_b_proj")
CONVS = ("q_conv1d", "k_conv1d", "v_conv1d")
HC_SCALES = {"self_attention_hyper_connection": "hc_attn_scale", "mlp_hyper_connection": "hc_ffn_scale"}
HC_ALPHAS = ("alpha_pre", "alpha_post", "alpha_res")


def old_layout(hidden: int, heads: int, head_dim: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The 13 KDA tensors of a GLM-5.3 checkpoint: name under ``self_attention.kda.`` -> (shape, dtype)."""
    size, bf16, fp32 = heads * head_dim, torch.bfloat16, torch.float32
    return {
        "q_proj.weight": ((size, hidden), bf16),
        "k_proj.weight": ((size, hidden), bf16),
        "v_proj.weight": ((size, hidden), bf16),
        "conv1d.weight": ((3 * size, 1, 4), bf16),
        "b_proj.weight": ((heads, hidden), bf16),
        "f_a_proj.weight": ((head_dim, hidden), bf16),
        "f_b_proj.weight": ((size, head_dim), bf16),
        "g_a_proj.weight": ((head_dim, hidden), bf16),
        "g_b_proj.weight": ((size, head_dim), bf16),
        "A_log": ((heads,), fp32),
        "dt_bias": ((size,), fp32),
        "o_norm.weight": ((head_dim,), bf16),
        "o_proj.weight": ((hidden, size), bf16),
    }


def random_old_tensors(sizes: dict, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=generator).to(dtype) for name, (shape, dtype) in old_layout(**sizes).items()
    }


def expected_local(old: dict[str, torch.Tensor], rank: int, world: int) -> dict[str, torch.Tensor]:
    """The runtime tensors of TP rank ``rank`` (names under ``self_attention.``) from the old full tensors."""
    local = {f"linear_attn.{name}.weight": old[f"{name}.weight"].chunk(world)[rank] for name in SHARDED_ROWS}
    for part, name in zip(old["conv1d.weight"].chunk(3), CONVS, strict=True):
        local[f"linear_attn.{name}.weight"] = part.chunk(world)[rank]
    local["linear_attn.A_log"] = old["A_log"].chunk(world)[rank]
    local["linear_attn.dt_bias"] = old["dt_bias"].chunk(world)[rank]
    local["linear_attn.f_a_proj.weight"] = old["f_a_proj.weight"]
    local["linear_attn.g_a_proj.weight"] = old["g_a_proj.weight"]
    local["linear_attn.norm.weight"] = old["o_norm.weight"]
    local["linear_attn.out_proj.weight"] = old["o_proj.weight"].chunk(world, dim=1)[rank]
    return local


def expected_hf(old: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    """The HF tensors of the old layout (the old converter split the packed conv in 3)."""
    hf = {f"model.layers.{layer}.self_attn.{name}": t for name, t in old.items() if name != "conv1d.weight"}
    for part, name in zip(old["conv1d.weight"].chunk(3), CONVS, strict=True):
        hf[f"model.layers.{layer}.self_attn.{name}.weight"] = part
    return hf


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup(rank: int, world: int, port: int, sizes: dict):
    """gloo on CPU, the CUDA calls of the layer constructor made CPU-safe, and a stub HF config."""
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    # On a GPU host the layer constructor asks for the device capability of "cpu" and fails.
    torch.cuda.is_available = lambda: False
    torch.cuda.current_device = lambda: "cpu"
    torch.cuda.synchronize = lambda *args, **kwargs: None  # dist_checkpointing syncs around save and load
    from megatron.core.transformer import TransformerConfig

    from miles_plugins.models import linear_attn
    from miles_plugins.models.glm5_next import kda

    linear_attn.get_cuda_rng_tracker = lambda: SimpleNamespace(fork=contextlib.nullcontext)
    text_config = SimpleNamespace(
        hidden_size=sizes["hidden"],
        rms_norm_eps=1e-5,
        linear_attn_config={"num_heads": sizes["heads"], "head_dim": sizes["head_dim"], "gate_lower_bound": -5.0},
    )
    kda.load_hf_config = lambda path: text_config
    singletons = [dist.new_group([r]) for r in range(world)]
    pg = SimpleNamespace(tp=dist.group.WORLD, cp=singletons[rank])
    config = TransformerConfig(
        num_layers=1,
        hidden_size=sizes["hidden"],
        num_attention_heads=sizes["heads"],
        tensor_model_parallel_size=world,
        sequence_parallel=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    metadata = {"non_homogeneous_layers": True, "dp_cp_group": singletons[rank]}

    def build():
        args = Namespace(hf_checkpoint="unused", allgather_cp=False)
        return kda.Glm5NextKDAAttention(args, config, layer_number=4, pg_collection=pg)

    return build, metadata


def _sharded(layer, metadata):
    return layer.sharded_state_dict(prefix=PREFIX, metadata=dict(metadata))


def _load(layer, metadata, ckpt_dir: str) -> None:
    """Load the way Megatron loads the model: the sharded state dict, then ``load_state_dict``."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.validation import StrictHandling

    loaded = dist_checkpointing.load(_sharded(layer, metadata), ckpt_dir, strict=StrictHandling.ASSUME_OK_UNEXPECTED)
    layer.load_state_dict({key[len(PREFIX) :]: value for key, value in loaded.items()}, strict=True)


def _old_layout_sharded(tensors: dict[str, torch.Tensor], rank: int) -> dict:
    """What the replicated layer saved: each tensor one chunk, the same on every rank."""
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    return {
        f"{PREFIX}kda.{name}": ShardedTensor.from_rank_offsets(
            f"{PREFIX}kda.{name}", t.clone(), replica_id=(0, rank, 0)
        )
        for name, t in tensors.items()
    }


def _gather_hf(layer, layer_idx: int) -> dict[str, torch.Tensor]:
    """The production weight-sync gather and the GLM-5.3 converter over every KDA parameter."""
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes

    from miles.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf
    from miles.backends.megatron_utils.update_weight.common import all_gather_params_async
    from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
    from miles.utils.ft_utils.process_group_utils import GroupInfo
    from miles.utils.types import ParamInfo

    one = GroupInfo(rank=0, size=1, group=None)
    tp = GroupInfo(rank=dist.get_rank(), size=dist.get_world_size(), group=dist.group.WORLD)
    set_parallel_state(
        ParallelState(intra_dp=one, intra_dp_cp=one, cp=one, tp=tp, pp=one, ep=one, etp=one, indep_dp=one)
    )
    infos_and_params = []
    for name, param in layer.named_parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        attrs = {
            "tensor_model_parallel": param.tensor_model_parallel,
            "partition_dim": param.partition_dim,
            "partition_stride": param.partition_stride,
            "parallel_mode": getattr(param, "parallel_mode", None),
        }
        full_name = f"module.module.decoder.layers.{layer_idx}.self_attention.{name}"
        info = ParamInfo(
            name=full_name, dtype=param.dtype, shape=param.shape, attrs=attrs, size=param.numel(), src_rank=0
        )
        # HfWeightIteratorDirect builds the same fresh Parameter with the ParamInfo attributes.
        copy = torch.nn.Parameter(param.detach().clone(), requires_grad=False)
        for key, value in attrs.items():
            setattr(copy, key, value)
        infos_and_params.append((info, copy))
    gathered = all_gather_params_async(Namespace(swiglu=True), infos_and_params)
    hf = {}
    for (info, _), tensor in zip(infos_and_params, gathered, strict=True):
        for hf_name, hf_tensor in convert_glm5_next_to_hf(Namespace(), info.name, tensor):
            assert hf_name not in hf, hf_name
            hf[hf_name] = hf_tensor
    return hf


def _round_trip_worker(rank: int, world: int, port: int, ckpt_root: str) -> None:
    from megatron.core import dist_checkpointing

    build, metadata = _setup(rank, world, port, SMALL)
    old = random_old_tensors(SMALL, seed=0)

    # 1. An old-layout save loads into the sharded layer, shard by shard.
    old_dir = f"{ckpt_root}/old"
    dist_checkpointing.save(_old_layout_sharded(old, rank), old_dir)
    dist.barrier()
    layer = build()
    _load(layer, metadata, old_dir)
    local = dict(layer.named_parameters())
    want = expected_local(old, rank, world)
    assert sorted(local) == sorted(want)
    bad = [name for name, t in want.items() if not (local[name].dtype == t.dtype and torch.equal(local[name], t))]
    assert not bad, f"rank {rank}: old save -> new layer mismatch {bad}"
    assert layer.linear_attn.A_log.keep_in_fp32 and layer.linear_attn.dt_bias.keep_in_fp32
    assert not [n for n, p in layer.named_parameters() if getattr(p, "sequence_parallel", False)]

    # 2. The weight sync gives the HF tensors of the old layout.
    hf = _gather_hf(layer, layer_idx=3)
    want_hf = expected_hf(old, layer=3)
    assert sorted(hf) == sorted(want_hf)
    bad = [n for n, t in want_hf.items() if not torch.equal(hf[n], t)]
    assert not bad, f"rank {rank}: weight sync mismatch {bad}"

    # 3. A save of the layer keeps the old keys and shapes; it loads into the layer and the old layout.
    new_dir = f"{ckpt_root}/new"
    dist_checkpointing.save(_sharded(layer, metadata), new_dir)
    dist.barrier()
    fresh = build()
    _load(fresh, metadata, new_dir)
    fresh_params = dict(fresh.named_parameters())
    bad = [n for n, p in layer.named_parameters() if not torch.equal(fresh_params[n], p)]
    assert not bad, f"rank {rank}: new save -> new layer mismatch {bad}"
    zeros = {name: torch.zeros_like(t) for name, t in old.items()}
    rollback = dist_checkpointing.load(_old_layout_sharded(zeros, rank), new_dir)
    bad = [name for name, t in old.items() if not torch.equal(rollback[f"{PREFIX}kda.{name}"], t)]
    assert not bad, f"rank {rank}: new save -> old layout mismatch {bad}"
    dist.destroy_process_group()


def _layout_worker(rank: int, world: int, port: int, sizes: dict) -> None:
    build, metadata = _setup(rank, world, port, sizes)
    layer = build()
    sharded = _sharded(layer, metadata)
    keys = {}
    for sh in sharded.values():
        shape, dtype = keys.setdefault(sh.key, (tuple(sh.global_shape), sh.data.dtype))
        assert (shape, dtype) == (tuple(sh.global_shape), sh.data.dtype), sh.key
    want = {f"{PREFIX}kda.{name}": spec for name, spec in old_layout(**sizes).items()}
    assert keys == want, f"rank {rank}: {keys} != {want}"
    # The three conv parts of this rank are 3 of the 3 * world fragments of the packed conv.
    conv = sorted(sh.global_offset[0] for sh in sharded.values() if sh.key.endswith("kda.conv1d.weight"))
    rows = sizes["heads"] * sizes["head_dim"] // world
    assert conv == [part * world * rows + rank * rows for part in range(3)]
    dist.destroy_process_group()


def test_old_save_loads_bit_exact_and_new_save_round_trips(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    mp.spawn(_round_trip_worker, args=(WORLD, _free_port(), str(tmp_path)), nprocs=WORLD, join=True)


def test_checkpoint_keys_shapes_and_dtypes_at_real_sizes():
    """The GLM-5.3-Flash checkpoint table (kdatp design, section 1) at TP 2."""
    mp.spawn(_layout_worker, args=(2, _free_port(), REAL), nprocs=2, join=True)


def test_mbridge_names_match_the_weight_sync_and_tp_split_restores():
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.glm5_next import Glm5NextBridge

    bridge = object.__new__(Glm5NextBridge)
    bridge.mpu = SimpleNamespace(tp_size=WORLD, etp_size=1)
    old = random_old_tensors(SMALL, seed=1)
    hf = expected_hf(old, layer=3)
    per_rank = [expected_local(old, rank, WORLD) for rank in range(WORLD)]
    for name, rank0 in per_rank[0].items():
        megatron_name = f"decoder.layers.3.self_attention.{name}"
        hf_name = _weight_sync_hf_name(name)
        assert bridge._weight_name_mapping_mcore_to_hf(megatron_name) == [
            hf_name.replace("model.", "model.language_model.", 1)
        ], name
        full = hf[hf_name]
        param = torch.nn.Parameter(torch.empty_like(rank0), requires_grad=False)
        param.tensor_model_parallel = rank0.shape != full.shape
        param.partition_dim = 1 if name.endswith("out_proj.weight") else 0
        splits = bridge._weight_split_across_tp(megatron_name, full, param, WORLD)
        assert all(torch.equal(split, want[name]) for split, want in zip(splits, per_rank, strict=True)), name
        if param.tensor_model_parallel:
            assert torch.equal(bridge._weight_merge_across_tp(megatron_name, list(splits), param), full), name


def _weight_sync_hf_name(name: str) -> str:
    from miles.backends.megatron_utils.megatron_to_hf.glm5_next import convert_glm5_next_to_hf

    megatron_name = f"module.module.decoder.layers.3.self_attention.{name}"
    [(hf_name, _)] = convert_glm5_next_to_hf(Namespace(), megatron_name, torch.empty(1))
    return hf_name


def _stored_dcp(ckpt_dir: str, layer: int) -> dict[str, torch.Tensor]:
    """Save a torch_dist checkpoint of one layer with the stored keys. Return the HF tensors of the weight sync.

    The checkpoint holds the 13 KDA tensors under ``self_attention.kda.`` and the three alphas of each
    hyper-connection site. The weight sync emits the alphas of a site as one ``hc_*_scale`` tensor.
    """
    import torch.distributed.checkpoint as dist_cp

    old = random_old_tensors(SMALL, seed=2)
    prefix = f"decoder.layers.{layer}."
    state = {f"{prefix}self_attention.kda.{name}": t for name, t in old.items()}
    want = expected_hf(old, layer)
    generator = torch.Generator().manual_seed(3)
    for site, scale in HC_SCALES.items():
        alphas = [torch.randn(1, generator=generator) for _ in HC_ALPHAS]
        state.update({f"{prefix}{site}.{alpha}": t for alpha, t in zip(HC_ALPHAS, alphas, strict=True)})
        want[f"model.layers.{layer}.{scale}"] = torch.cat(alphas)
    dist_cp.save(state, storage_writer=dist_cp.FileSystemWriter(ckpt_dir), no_dist=True)
    return want


def _read_safetensors(out_dir) -> dict[str, torch.Tensor]:
    import safetensors.torch

    tensors = {}
    for path in sorted(out_dir.glob("*.safetensors")):
        shard = safetensors.torch.load_file(str(path))
        assert not set(shard) & set(tensors), path
        tensors.update(shard)
    return tensors


def _assert_bit_exact(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> None:
    assert sorted(got) == sorted(want)
    bad = [n for n, t in want.items() if not (got[n].dtype == t.dtype and torch.equal(got[n], t))]
    assert not bad, f"mismatch {bad}"


def test_offline_tools_convert_the_stored_keys(tmp_path, monkeypatch):
    """The two DCP-to-HF tools write the HF tensors of the weight sync from a checkpoint with the stored keys.

    The Ray tool runs each task in one actor of many, and the alpha buffer is per process. So each task that
    starts a ``hc_*_scale`` tensor also completes it.
    """
    import torch.distributed.checkpoint as dist_cp

    from miles.backends.megatron_utils.megatron_to_hf import glm5_next

    monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)  # the tools replace it at import
    from tools import convert_torch_dist_to_hf as tool
    from tools import convert_torch_dist_to_hf_ray as ray_tool

    ckpt = str(tmp_path / "dcp")
    want = _stored_dcp(ckpt, layer=3)
    # The fields of the common.pt args that the tools read for these keys.
    megatron_args = Namespace(num_layers=4, vocab_size=None)
    assert not glm5_next._hc_scale_buffers

    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=tool.WrappedStorageReader(ckpt),
        planner=tool.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    tool.save_tensors(megatron_args, "glm5_next", state_dict, str(tmp_path / "hf"), chunk_size=2**30)
    _assert_bit_exact(_read_safetensors(tmp_path / "hf"), want)
    assert not glm5_next._hc_scale_buffers

    metadata = ray_tool.WrappedStorageReader(ckpt).read_metadata()
    tensor_metadata = ray_tool.tensor_metadata_from_checkpoint_metadata(metadata)
    tasks = ray_tool.plan_conversion_tasks(tensor_metadata, metadata, q_lora_rank=None, task_group_bytes=0)
    staging = tmp_path / "hf_ray"
    staging.mkdir()
    for task in tasks:
        prepared = ray_tool.prepare_whole_source_task_tensors(task, ckpt, megatron_args, "glm5_next", metadata)
        assert not glm5_next._hc_scale_buffers, f"task {task.keys} leaves a hc_*_scale incomplete"
        ray_tool.write_prepared_tensor_groups(
            str(staging), task.task_id, prepared.groups, megatron_args, None, 2**30, None
        )
    _assert_bit_exact(_read_safetensors(staging), want)
