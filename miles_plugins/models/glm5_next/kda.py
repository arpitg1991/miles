"""GLM-5.3-Flash KDA layers on the shared head-sharded delta-rule layer (``miles_plugins.models.linear_attn``).

Tensor parallelism splits the 64 KDA heads, 8 heads per rank at TP 8. The Megatron checkpoint keeps the
layout of the replicated layer that wrote the base and the r4x checkpoints: keys under
``self_attention.kda.``, the names ``o_norm`` and ``o_proj``, one packed ``conv1d.weight`` ``[q; k; v]``,
and the full global shapes. ``sharded_state_dict`` maps the runtime names onto those keys. Thus an old
DCP loads into this layer, and a save of this layer loads into the old replicated layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region

from miles.kernels.attention.delta_rule import DeltaRuleHeads, KimiDeltaRule
from miles.utils.hf_config import load_hf_config
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer, Projections

# The per-tensor convs of the shared layer, in the row order of the packed checkpoint conv [q; k; v].
KDA_CONVS = ("q_conv1d", "k_conv1d", "v_conv1d")
# Runtime name prefix -> checkpoint name prefix under ``self_attention.``. The first match wins.
KDA_CHECKPOINT_PREFIXES = {
    "linear_attn.norm.": "kda.o_norm.",
    "linear_attn.out_proj.": "kda.o_proj.",
    "linear_attn.": "kda.",
}


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


class Glm5NextDeltaAttention(KimiDeltaAttention):
    """The GLM-5.3 KDA heads of this rank. Kimi-K3 KDA with two differences:

    - The output gate is low rank, ``g_b_proj(g_a_proj(x))``, like the forget gate. ``g_a_proj`` and
      ``f_a_proj`` are replicated and feed head-sharded linears, so their weights pass the TP copy op
      (the backward sums their gradients across TP).
    - The conv weights stay bf16, as in the GLM-5.3 checkpoints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.local.key_dim == self.local.value_dim, "the packed checkpoint conv needs equal q/k/v widths"
        # The checkpoint layout did not change, so GLM-5.3 DCPs carry no layout marker.
        self.register_buffer("weight_layout_version", self.weight_layout_version, persistent=False)

    def _replicated_linear(self, input_size: int, output_size: int) -> nn.Linear:
        linear = nn.Linear(
            input_size, output_size, bias=False, device=torch.cuda.current_device(), dtype=self.config.params_dtype
        )
        self.config.init_method(linear.weight)
        return linear

    def _build_projections(self):
        hidden, head_dim, local = self.config.hidden_size, self.heads.head_v_dim, self.local
        self.q_proj = self.sharded_linear("q_proj", hidden, local.key_dim)
        self.k_proj = self.sharded_linear("k_proj", hidden, local.key_dim)
        self.v_proj = self.sharded_linear("v_proj", hidden, local.value_dim)
        self.b_proj = self.sharded_linear("b_proj", hidden, local.num_v_heads)
        self.f_a_proj = self._replicated_linear(hidden, head_dim)
        self.f_b_proj = self.sharded_linear("f_b_proj", head_dim, local.value_dim)
        self.g_a_proj = self._replicated_linear(hidden, head_dim)
        self.g_b_proj = self.sharded_linear("g_b_proj", head_dim, local.value_dim)

    def _build_convolutions(self):
        local = self.local
        self.q_conv1d = self.sharded_conv(local.key_dim)
        self.k_conv1d = self.sharded_conv(local.key_dim)
        self.v_conv1d = self.sharded_conv(local.value_dim)

    def project(self, x):
        f_a_weight = copy_to_tensor_model_parallel_region(self.f_a_proj.weight, group=self.tp_group)
        g_a_weight = copy_to_tensor_model_parallel_region(self.g_a_proj.weight, group=self.tp_group)
        return Projections(
            (self.q_proj(x), self.k_proj(x), self.v_proj(x)),
            self.g_b_proj(F.linear(x, g_a_weight)),
            self.b_proj(x),
            self.f_b_proj(F.linear(x, f_a_weight)),
        )

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        """Save the three conv shards of this rank into the one packed checkpoint tensor ``conv1d.weight``.

        Rank ``r`` holds rows ``r`` of each part of ``[q; k; v]``. Part ``i`` is fragment ``i * tp + r`` of
        ``3 * tp`` on axis 0, so the saved tensor has the global shape and the row order of the packed conv.
        """
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        tp_rank, tp_size = self.tp_group.rank(), self.tp_group.size()
        axis = len(sharded_offsets)
        for part, name in enumerate(KDA_CONVS):
            key = f"{prefix}{name}.weight"
            entry = sharded[key]
            sharded[key] = ShardedTensor.from_rank_offsets(
                f"{prefix}conv1d.weight",
                entry.data,
                *sharded_offsets,
                (axis, part * tp_size + tp_rank, len(KDA_CONVS) * tp_size),
                replica_id=entry.replica_id,
                prepend_axis_num=axis,
            )
        return sharded


class Glm5NextKDAAttention(LinearAttentionLayer):
    """The GLM-5.3 KDA ``self_attention`` on the shared head-sharded layer. The input norm belongs to the
    transformer layer, so this layer has none."""

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        del layer_number, cp_comm_type, name
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        text_config = _get_text_config(load_hf_config(args.hf_checkpoint))
        fields = _linear_attn_fields(text_config)
        heads = DeltaRuleHeads(
            num_k_heads=fields["num_heads"],
            num_v_heads=fields["num_heads"],
            head_k_dim=fields["head_dim"],
            head_v_dim=fields["head_dim"],
        )
        core = Glm5NextDeltaAttention(
            config,
            heads,
            KimiDeltaRule(fields["gate_lower_bound"]),
            fields["conv_kernel_size"],
            text_config.rms_norm_eps,
            pg_collection.tp,
        )
        super().__init__(config, core, nn.Identity(), pg_collection, allgather_cp=args.allgather_cp)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None):
        """The state-dict keys keep the runtime names. The checkpoint keys keep the names of the old layer."""
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        apply_prefix_mapping(
            sharded, {f"{prefix}{old}": f"{prefix}{new}" for old, new in KDA_CHECKPOINT_PREFIXES.items()}
        )
        return sharded
