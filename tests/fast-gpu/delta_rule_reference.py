"""Replicated, HF-layout references for the head-sharded delta-rule layers, and helpers to load their
weights into a :class:`LinearAttentionLayer`.

Ported from radixark/miles ``ff6193f26`` (PR #3609 stack). The GDN (Qwen3.5 / Qwen3-Next) references are
not ported: their head-sharded layers come with PR #3610, which this fork does not take. The GLM-5.3
reference is the replicated layer that trained the GLM-5.3 checkpoints of this fork.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda
from fla.ops.kda.gate import fused_kda_gate
from megatron.core.process_groups_config import ProcessGroupCollection

from miles.kernels.attention.delta_rule import DeltaRuleHeads, KimiDeltaRule
from miles_plugins.models.glm5_next.kda import Glm5NextDeltaAttention
from miles_plugins.models.linear_attn import KimiDeltaAttention, LinearAttentionLayer

CONV = 4
EPS = 1e-6
KDA_LOWER_BOUND = -5.0


class ReplicatedKDA(nn.Module):
    """Pre-sharding KDA in the Kimi-K3 HF layout."""

    def __init__(self, hidden: int, heads: DeltaRuleHeads, dtype):
        super().__init__()
        self.heads = heads
        size, d = heads.value_dim, heads.head_v_dim
        self.q_proj = nn.Linear(hidden, size, bias=False)
        self.k_proj = nn.Linear(hidden, size, bias=False)
        self.v_proj = nn.Linear(hidden, size, bias=False)
        self.q_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(size, CONV, bias=False, activation="silu")
        self.f_a_proj = nn.Linear(hidden, d, bias=False)
        self.f_b_proj = nn.Linear(d, size, bias=False)
        self.b_proj = nn.Linear(hidden, heads.num_v_heads, bias=False)
        self.g_proj = nn.Linear(hidden, size, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads.num_v_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.rand(size))
        self.o_norm = FusedRMSNormGated(d, eps=EPS, activation="sigmoid")
        self.o_proj = nn.Linear(size, hidden, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()

    def hf_conv(self):
        return torch.cat([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight])

    def forward(self, x, cu_seqlens):
        bsz, seq_len, _ = x.shape
        h, d = self.heads.num_v_heads, self.heads.head_v_dim
        q, _ = self.q_conv1d(self.q_proj(x), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(self.k_proj(x), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(self.v_proj(x), cu_seqlens=cu_seqlens)
        out, _ = chunk_kda(
            q=q.view(bsz, seq_len, h, d),
            k=k.view(bsz, seq_len, h, d),
            v=v.view(bsz, seq_len, h, d),
            g=self.f_b_proj(self.f_a_proj(x)).view(bsz, seq_len, h, d),
            beta=self.b_proj(x).float().sigmoid(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=KDA_LOWER_BOUND,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        out = self.o_norm(out.reshape(-1, d), self.g_proj(x).reshape(-1, d))
        return self.o_proj(out.reshape(bsz, seq_len, -1))


class ReplicatedGlm5NextKDA(nn.Module):
    """The GLM-5.3 KDA layer of this fork before the shared layer: all heads on each rank, one packed
    ``[q; k; v]`` conv, a low-rank output gate, and the names and shapes of the GLM-5.3 checkpoints.

    ``gate`` selects the kernel call. ``"outside"`` is the old trainer: ``fused_kda_gate``, then
    ``chunk_kda`` on the decay. ``"kernel"`` is the shared layer (``KimiDeltaRule``): the gate inside
    ``chunk_kda`` with the safe-gate path, as SGLang runs GLM-5.3. Both compute the same function.
    """

    def __init__(self, hidden: int, heads: DeltaRuleHeads, dtype, gate: str = "outside", eps: float = EPS):
        super().__init__()
        assert gate in ("outside", "kernel")
        self.heads, self.gate = heads, gate
        size, d = heads.value_dim, heads.head_v_dim
        self.q_proj = nn.Linear(hidden, size, bias=False)
        self.k_proj = nn.Linear(hidden, size, bias=False)
        self.v_proj = nn.Linear(hidden, size, bias=False)
        self.conv1d = ShortConvolution(hidden_size=3 * size, kernel_size=CONV, bias=False, activation="silu")
        self.b_proj = nn.Linear(hidden, heads.num_v_heads, bias=False)
        self.f_a_proj = nn.Linear(hidden, d, bias=False)
        self.f_b_proj = nn.Linear(d, size, bias=False)
        self.g_a_proj = nn.Linear(hidden, d, bias=False)
        self.g_b_proj = nn.Linear(d, size, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads.num_v_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.rand(size))
        self.o_norm = FusedRMSNormGated(d, eps=eps, activation="sigmoid")
        self.o_proj = nn.Linear(size, hidden, bias=False)
        self.to(dtype=dtype)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()

    def forward(self, x, cu_seqlens):
        bsz, seq_len, _ = x.shape
        h, d = self.heads.num_v_heads, self.heads.head_v_dim
        size = self.heads.value_dim
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        mixed, _ = self.conv1d(x=qkv, cu_seqlens=cu_seqlens)
        q, k, v = (t.unflatten(-1, (h, d)) for t in mixed.split([size] * 3, dim=-1))
        beta = torch.sigmoid(self.b_proj(x).float())
        forget = self.f_b_proj(self.f_a_proj(x)).unflatten(-1, (h, d))
        if self.gate == "outside":
            g = fused_kda_gate(forget, self.A_log, self.dt_bias, lower_bound=KDA_LOWER_BOUND)
            out, _ = chunk_kda(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            out, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=forget,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=KDA_LOWER_BOUND,
                transpose_state_layout=True,
                cu_seqlens=cu_seqlens,
            )
        gate = self.g_b_proj(self.g_a_proj(x))
        out = self.o_norm(out.reshape(-1, d), gate.reshape(-1, d))
        return self.o_proj(out.reshape(bsz, seq_len, -1))

    def hf_conv(self):
        return self.conv1d.weight


def sharded_projections(ref, grad: bool = False) -> dict[str, torch.Tensor]:
    """Full (all-head) Megatron tensor of every head-sharded projection, from the reference's HF-layout
    weights or their gradients: what the bridges load."""

    def w(module):
        return module.weight.grad if grad else module.weight

    names = ("q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj")
    names += ("g_b_proj",) if isinstance(ref, ReplicatedGlm5NextKDA) else ("g_proj",)
    return {name: w(getattr(ref, name)) for name in names}


def replicated_projections(ref, grad: bool = False) -> dict[str, torch.Tensor]:
    """The projections that every TP rank holds whole."""

    def w(module):
        return module.weight.grad if grad else module.weight

    names = ("f_a_proj", "g_a_proj") if isinstance(ref, ReplicatedGlm5NextKDA) else ("f_a_proj",)
    return {name: w(getattr(ref, name)) for name in names}


def conv_of(ref, grad: bool = False) -> dict[str, torch.Tensor]:
    """Full Megatron conv weight of each core conv: one per q, k and v for KDA."""

    def w(module):
        return module.weight.grad if grad else module.weight

    if isinstance(ref, ReplicatedGlm5NextKDA):
        return dict(zip(("q_conv1d", "k_conv1d", "v_conv1d"), w(ref.conv1d).chunk(3), strict=True))
    return {name: w(getattr(ref, name)) for name in ("q_conv1d", "k_conv1d", "v_conv1d")}


def shard(full: torch.Tensor, dim: int, group) -> torch.Tensor:
    return full.chunk(group.size(), dim=dim)[group.rank()].contiguous()


def gather(local: torch.Tensor, dim: int, group) -> torch.Tensor:
    parts = [torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts, local.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def build_layer(ref, config, allgather_cp: bool = True) -> LinearAttentionLayer:
    """A head-sharded layer holding this TP rank's shard of ``ref``'s weights, identity input norm."""
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    tp = pg.tp
    core_cls = Glm5NextDeltaAttention if isinstance(ref, ReplicatedGlm5NextKDA) else KimiDeltaAttention
    core = core_cls(config, ref.heads, KimiDeltaRule(KDA_LOWER_BOUND), CONV, EPS, tp)
    with torch.no_grad():
        for name, full in sharded_projections(ref).items():
            getattr(core, name).weight.copy_(shard(full, 0, tp))
        for name, full in conv_of(ref).items():
            getattr(core, name).weight.copy_(shard(full, 0, tp))
        for name, full in replicated_projections(ref).items():
            getattr(core, name).weight.copy_(full)
        core.A_log.copy_(shard(ref.A_log, 0, tp))
        core.dt_bias.copy_(shard(ref.dt_bias, 0, tp))
        core.norm.weight.copy_(ref.o_norm.weight)
        core.out_proj.weight.copy_(shard(ref.o_proj.weight, 1, tp))
    return LinearAttentionLayer(config, core, nn.Identity(), pg, allgather_cp=allgather_cp)


def packed(cu_seqlens):
    return SimpleNamespace(cu_seqlens_q=cu_seqlens)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()
