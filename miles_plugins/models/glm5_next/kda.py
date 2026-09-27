import copy

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor, ShardedTensorFactory
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TERowParallelLinear
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None
    fused_kda_gate = None

from miles_plugins.models.cp_utils import build_gdn_cp_context
from miles_plugins.models.hf_attention import HuggingfaceAttention


def _get_text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_fields(text_config) -> dict:
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if not isinstance(linear_attn_config, dict):
        linear_attn_config = {}

    def field(key, attr, default):
        value = linear_attn_config.get(key)
        if value is None:
            value = getattr(text_config, attr, default)
        return value

    gate_lower_bound = field("gate_lower_bound", "gate_lower_bound", None)
    if gate_lower_bound is None:
        gate_lower_bound = getattr(text_config, "linear_lower_bound", None)
    if gate_lower_bound is None:
        raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
    return dict(
        num_heads=int(field("num_heads", "linear_num_heads", 64)),
        head_dim=int(field("head_dim", "linear_head_dim", 128)),
        conv_kernel_size=int(field("short_conv_kernel_size", "linear_conv_kernel_dim", 4)),
        gate_lower_bound=float(gate_lower_bound),
    )


# The packed conv weight holds three parts, [q; k; v], of projection_size rows each.
_KDA_CONV_PARTS = 3


def _require_fla() -> None:
    if ShortConvolution is None or chunk_kda is None:
        raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")


class _Glm5NextKDAMath:
    """The KDA forward pass, shared by the replicated and the tensor-parallel module.

    The subclass sets ``local_num_heads`` and ``local_projection_size`` (the
    heads of this rank) and ``_linear`` (one projection call).
    """

    def _init_sizes(self, hidden_size, num_heads, head_dim, conv_kernel_size, gate_lower_bound, tp_size) -> None:
        assert num_heads % tp_size == 0, f"KDA num_heads={num_heads} is not divisible by tp_size={tp_size}"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.local_num_heads = num_heads // tp_size
        self.local_projection_size = self.local_num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

    def _linear(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return module(x)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):
        cp_context = build_gdn_cp_context(self, cu_seqlens, hidden_states.device)

        mixed_qkv = torch.cat(
            (
                self._linear(self.q_proj, hidden_states),
                self._linear(self.k_proj, hidden_states),
                self._linear(self.v_proj, hidden_states),
            ),
            dim=-1,
        )
        conv_cu_seqlens = cp_context.cu_seqlens if cp_context is not None else cu_seqlens
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=conv_cu_seqlens,
            cp_context=cp_context,
        )
        query, key, value = torch.split(mixed_qkv, [self.local_projection_size] * _KDA_CONV_PARTS, dim=-1)
        query = query.unflatten(-1, (self.local_num_heads, self.head_dim))
        key = key.unflatten(-1, (self.local_num_heads, self.head_dim))
        value = value.unflatten(-1, (self.local_num_heads, self.head_dim))

        beta = torch.sigmoid(self._linear(self.b_proj, hidden_states).float())
        forget = self._linear(self.f_b_proj, self._linear(self.f_a_proj, hidden_states))
        g = fused_kda_gate(
            forget.unflatten(-1, (self.local_num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )

        if cp_context is not None:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            core_attn_out, _ = chunk_kda(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        norm_gate = self._linear(self.g_b_proj, self._linear(self.g_a_proj, hidden_states))
        out_shape = core_attn_out.shape
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1)
        return self._linear(self.o_proj, core_attn_out)


class Glm5NextKDA(_Glm5NextKDAMath, nn.Module):
    """GLM-5.3 KDA with all heads on every tensor-parallel rank (the default)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        _require_fla()
        self._init_sizes(hidden_size, num_heads, head_dim, conv_kernel_size, gate_lower_bound, tp_size=1)

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = mark_keep_in_fp32(nn.Parameter(torch.zeros(num_heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.projection_size, dtype=torch.float32)))
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)


class Glm5NextKDATensorParallel(_Glm5NextKDAMath, MegatronModule):
    """GLM-5.3 KDA with the heads split across the tensor-parallel ranks.

    Port of upstream miles Kimi-K3 ``KimiK3Attention._init_kda`` (PR #1825).
    Each rank holds ``num_heads / tp`` heads. The parameter names and the
    global shapes are the same as in ``Glm5NextKDA``, so each checkpoint
    loads in both modules.

    - Column-parallel (rows split): q, k, v, b, f_b, g_b.
    - Row-parallel (columns split): o_proj. Its output all-reduce sums the heads.
    - Duplicated: f_a, g_a. Their output gradient comes from the column-parallel
      f_b and g_b, which all-reduce it, so each rank gets the full weight gradient.
    - Head-split: A_log, dt_bias, and the packed conv (one head slice of q, k
      and v per rank, partition stride 3).
    - Replicated: o_norm. Each rank sees only its heads, so its gradient is
      summed across the ranks. The sum is the sequence-parallel all-reduce in
      ``finalize_model_grads``.

    The input arrives already gathered over the sequence (``HuggingfaceAttention``),
    so the linears run with ``sequence_parallel`` off.
    """

    def __init__(
        self,
        config,
        tp_group,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
    ):
        super().__init__(config=config)
        _require_fla()
        self.tp_group = tp_group
        tp_size = tp_group.size()
        # o_norm gets its cross-rank gradient sum only from the sequence-parallel all-reduce.
        assert config.sequence_parallel or tp_size == 1, "GLM-5.3 KDA tensor parallelism requires sequence_parallel"
        self._init_sizes(hidden_size, num_heads, head_dim, conv_kernel_size, gate_lower_bound, tp_size)

        # A copy with sequence_parallel off. With the original config, the duplicated
        # f_a/g_a weights get the sequence_parallel mark and their full gradients
        # are summed tp_size times.
        linear_config = copy.copy(config)
        linear_config.sequence_parallel = False

        def column(input_size: int, output_size: int) -> TEColumnParallelLinear:
            return TEColumnParallelLinear(
                input_size,
                output_size,
                config=linear_config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_group=tp_group,
            )

        def duplicated(input_size: int, output_size: int) -> TELinear:
            return TELinear(
                input_size,
                output_size,
                parallel_mode="duplicated",
                config=linear_config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
            )

        # Same registration order as Glm5NextKDA.
        self.q_proj = column(hidden_size, self.projection_size)
        self.k_proj = column(hidden_size, self.projection_size)
        self.v_proj = column(hidden_size, self.projection_size)
        self.conv1d = ShortConvolution(
            hidden_size=_KDA_CONV_PARTS * self.local_projection_size,
            kernel_size=conv_kernel_size,
            bias=False,
            activation="silu",
        )
        set_tensor_model_parallel_attributes(self.conv1d.weight, True, 0, _KDA_CONV_PARTS)
        self.b_proj = column(hidden_size, num_heads)
        self.f_a_proj = duplicated(hidden_size, head_dim)
        self.f_b_proj = column(head_dim, self.projection_size)
        self.g_a_proj = duplicated(hidden_size, head_dim)
        self.g_b_proj = column(head_dim, self.projection_size)
        self.A_log = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.local_num_heads, dtype=torch.float32)))
        self.dt_bias = mark_keep_in_fp32(nn.Parameter(torch.zeros(self.local_projection_size, dtype=torch.float32)))
        set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
        set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_norm.weight.sequence_parallel = True
        self.o_proj = TERowParallelLinear(
            self.projection_size,
            hidden_size,
            config=linear_config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=tp_group,
        )

    def _linear(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output, bias = module(x)
        assert bias is None
        return output

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """The TE linears shard themselves; A_log, dt_bias and the conv are replaced here."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        dp_cp_group = metadata["dp_cp_group"]
        sharded_state_dict.update(
            make_sharded_tensors_for_checkpoint(
                {"A_log": self.A_log, "dt_bias": self.dt_bias},
                prefix,
                {"A_log": 0, "dt_bias": 0},
                sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=dp_cp_group,
            )
        )
        conv_key = f"{prefix}conv1d.weight"
        sharded_state_dict[conv_key] = tp_strided_sharded_factory(
            make_tp_sharded_tensor_for_checkpoint(
                self.conv1d.weight,
                conv_key,
                0,
                prepend_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=dp_cp_group,
            ),
            sharded_offsets,
            num_parts=_KDA_CONV_PARTS,
        )
        return sharded_state_dict


def tp_strided_sharded_factory(
    original_sh_ten: ShardedTensor, sharded_offsets: tuple, num_parts: int
) -> ShardedTensorFactory:
    """Save a ``num_parts``-way strided TP shard under its original key and global shape.

    The local tensor is ``[part_0 slice r; ...; part_{n-1} slice r]``. Each part
    is its own fragment ``i * tp + r`` of ``n * tp`` along axis 0, so the saved
    tensor keeps the layout of the unsharded weight. Megatron's
    ``apply_swiglu_sharded_factory`` does the same for 2 parts.
    """
    prepend_axis_num = len(sharded_offsets)
    local_rows = original_sh_ten.local_shape[0]
    assert original_sh_ten.global_offset[prepend_axis_num] % local_rows == 0
    rank_offset = original_sh_ten.global_offset[prepend_axis_num] // local_rows
    axis_frag = original_sh_ten.axis_fragmentations[prepend_axis_num]

    @torch.no_grad()
    def build_fn(key, t, replica_id, flattened_range):
        return [
            ShardedTensor.from_rank_offsets(
                key,
                part,
                *sharded_offsets,
                (prepend_axis_num, i * axis_frag + rank_offset, axis_frag * num_parts),
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            )
            for i, part in enumerate(torch.chunk(t, num_parts, dim=0))
        ]

    @torch.no_grad()
    def merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        build_fn,
        merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )


class Glm5NextKDAAttention(HuggingfaceAttention):

    hybrid_cp = True

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection, name=name)
        text_config = _get_text_config(self.hf_config)
        fields = _linear_attn_fields(text_config)
        kda_kwargs = dict(
            hidden_size=text_config.hidden_size,
            num_heads=fields["num_heads"],
            head_dim=fields["head_dim"],
            conv_kernel_size=fields["conv_kernel_size"],
            gate_lower_bound=fields["gate_lower_bound"],
            rms_norm_eps=text_config.rms_norm_eps,
        )
        if getattr(args, "glm5_next_kda_tp", False):
            tp_group = (
                pg_collection.tp if pg_collection is not None else parallel_state.get_tensor_model_parallel_group()
            )
            self.kda = Glm5NextKDATensorParallel(config=config, tp_group=tp_group, **kda_kwargs)
        else:
            self.kda = Glm5NextKDA(**kda_kwargs)

    def hf_forward(self, hidden_states, packed_seq_params):
        return self.kda(hidden_states, cu_seqlens=packed_seq_params.cu_seqlens_q)
