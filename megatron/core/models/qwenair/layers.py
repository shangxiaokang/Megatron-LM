# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir text layers with a differentiable single-rank reference path."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import QwenAirTextConfig
from .initialization import initialize_qwenair_expert_normal_


class QwenAirRMSNorm(nn.Module):
    """Zero-centered RMSNorm, optionally normalized per residual stream."""

    def __init__(self, width: int, eps: float, group_size: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        """Normalize in FP32 and apply the ``1 + weight`` scale."""
        original_dtype = x.dtype
        x_float = x.float()
        if self.group_size is not None:
            x_float = x_float.unflatten(-1, (-1, self.group_size))
        x_float = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            x_float = x_float.flatten(-2)
        return (x_float * (1 + self.weight.float())).to(original_dtype)


class QwenAirGatedRMSNorm(nn.Module):
    """GDN output RMSNorm with an affine weight and sigmoid gate."""

    def __init__(self, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.variance_epsilon = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        """Apply the HF order: FP32 norm, cast, weight, FP32 sigmoid gate."""
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.variance_epsilon)
        return ((normalized.to(dtype) * self.weight) * torch.sigmoid(gate.float())).to(dtype)


class QwenAirGatedResidual(nn.Module):
    """Four-stream Qwen read/inject connection, distinct from Sinkhorn mHC."""

    def __init__(self, config: QwenAirTextConfig, use_combine: bool = True) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        width = self.hc_count * self.hidden_size
        self.hc_norm = QwenAirRMSNorm(width, config.rms_norm_eps, group_size=self.hidden_size)
        self.input_mix_weight_down = nn.Linear(width, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, width, bias=False)
        self.block_inject_weight = nn.Linear(width, self.hc_count, bias=False) if use_combine else None

    def forward(self, hyper_input: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Read a single block input and optionally return stream injection gates."""
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError("QwenAir hyper input has the wrong width")
        normalized = self.hc_norm(hyper_input)
        weights = torch.sigmoid(
            self.input_mix_weight_up(F.silu(self.input_mix_weight_down(normalized) / self.hc_count))
        ).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (weights * normalized.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * torch.sigmoid(self.block_inject_weight(normalized) / self.hc_count)
        return mixed, hyper_input, injection


def inject_hyper_output(hyper_input: Tensor, block_output: Tensor, injection: Tensor) -> Tensor:
    """Broadcast a block output into the residual streams."""
    return hyper_input + (block_output.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)


def qwenair_rope(config: QwenAirTextConfig, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Create partial rotary embeddings, including Qwen's interleaved MRoPE."""
    if positions.ndim == 2:
        positions = positions.unsqueeze(0).expand(3, -1, -1)
    elif positions.ndim == 3 and positions.shape[0] == 1:
        positions = positions.expand(3, -1, -1)
    elif positions.ndim == 3 and positions.shape[0] == 4:
        # Qwen4-Exp carries one text-mask position axis before the three
        # rotary axes. The first axis does not participate in MRoPE.
        positions = positions[1:]
    if positions.ndim != 3 or positions.shape[0] != 3:
        raise ValueError("position_ids must be [batch, seq] or [3, batch, seq]")
    rotary_dim = config.rotary_dim
    inv_freq = 1 / (
        config.rope_theta
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=positions.device) / rotary_dim)
    )
    frequencies = positions.float().unsqueeze(-1) * inv_freq
    mixed = frequencies[0].clone()
    for axis in (1, 2):
        end = config.mrope_section[axis] * 3
        mixed[..., axis:end:3] = frequencies[axis, ..., axis:end:3]
    doubled = torch.cat((mixed, mixed), dim=-1)
    return doubled.cos().to(dtype), doubled.sin().to(dtype)


def apply_qwenair_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the leading coordinates of a [batch, seq, heads, dim] tensor."""
    rotary_dim = cos.shape[-1]
    rotated, plain = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotated.chunk(2, dim=-1)
    half_rotated = torch.cat((-second, first), dim=-1)
    result = rotated * cos.unsqueeze(2) + half_rotated * sin.unsqueeze(2)
    return torch.cat((result, plain), dim=-1)


class QwenAirGatedDeltaNet(nn.Module):
    """Differentiable recurrence with HF-compatible GDN parameter names.

    This path is intended for correctness and small training runs. The target
    262k context requires FLA/TE kernels and distributed activation handling.
    """

    def __init__(self, config: QwenAirTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.conv_kernel_size,
            groups=self.conv_dim, padding=0, bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0.01, 16).log_())
        self.norm = QwenAirGatedRMSNorm(self.head_v_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

    def forward(self, x: Tensor, token_mask: Tensor | None = None) -> Tensor:
        """Evaluate the causal delta-rule recurrence without a persistent cache."""
        batch, length, _ = x.shape
        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1)
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        qkv = F.silu(self.conv1d(F.pad(qkv, (self.conv_kernel_size - 1, 0))).transpose(1, 2))
        q, k, v = torch.split(qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.unflatten(-1, (self.num_k_heads, self.head_k_dim))
        k = k.unflatten(-1, (self.num_k_heads, self.head_k_dim))
        v = v.unflatten(-1, (self.num_v_heads, self.head_v_dim))
        z = self.in_proj_z(x).unflatten(-1, (self.num_v_heads, self.head_v_dim))
        beta = torch.sigmoid(self.in_proj_b(x).float())
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        repeat = self.num_v_heads // self.num_k_heads
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
        q = (q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)).float()
        k = (k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)).float()
        q = q * (1 / math.sqrt(self.head_k_dim))
        state = torch.zeros(
            batch, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            dtype=torch.float32, device=x.device,
        )
        outputs: list[Tensor] = []
        for step in range(length):
            key = k[:, step]
            state = state * decay[:, step].exp().unsqueeze(-1).unsqueeze(-1)
            prediction = (state * key.unsqueeze(-1)).sum(dim=-2)
            correction = (v[:, step].float() - prediction) * beta[:, step].unsqueeze(-1)
            state = state + key.unsqueeze(-1) * correction.unsqueeze(-2)
            outputs.append((state * q[:, step].unsqueeze(-1)).sum(dim=-2))
        result = torch.stack(outputs, dim=1).to(x.dtype)
        result = self.norm(result, z).flatten(-2)
        return self.out_proj(result)


class QwenAirMLP(nn.Module):
    """SwiGLU shared expert with Hugging Face parameter names."""

    def __init__(self, config: QwenAirTextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenAirExperts(nn.Module):
    """Packed routed experts, with optional local-only expert allocation."""

    def __init__(
        self,
        config: QwenAirTextConfig,
        num_local_experts: int | None = None,
        *,
        layer_idx: int = 0,
        first_global_expert: int = 0,
    ) -> None:
        super().__init__()
        self.intermediate_dim = config.moe_intermediate_size
        expert_count = config.num_experts if num_local_experts is None else num_local_experts
        if expert_count < 1 or expert_count > config.num_experts:
            raise ValueError("num_local_experts must be between 1 and num_experts")
        if first_global_expert < 0 or first_global_expert + expert_count > config.num_experts:
            raise ValueError("The local QwenAir expert range is outside the logical expert set")
        self.gate_up_proj = nn.Parameter(
            torch.empty(expert_count, 2 * self.intermediate_dim, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(expert_count, config.hidden_size, self.intermediate_dim)
        )
        initialize_qwenair_expert_normal_(
            self.gate_up_proj,
            base_seed=config.seed,
            namespace="qwenair.expert.gate_up",
            layer_idx=layer_idx,
            first_global_expert=first_global_expert,
            std=config.initializer_range,
        )
        initialize_qwenair_expert_normal_(
            self.down_proj,
            base_seed=config.seed,
            namespace="qwenair.expert.down",
            layer_idx=layer_idx,
            first_global_expert=first_global_expert,
            std=config.initializer_range,
        )

    def forward(self, hidden: Tensor, indices: Tensor, scores: Tensor) -> Tensor:
        """Dispatch selected tokens without materializing all expert activations."""
        result = torch.zeros_like(hidden)
        for expert in range(self.gate_up_proj.shape[0]):
            token_idx, slot_idx = torch.where(indices == expert)
            if token_idx.numel() == 0:
                continue
            source = hidden.index_select(0, token_idx)
            gate, up = F.linear(source, self.gate_up_proj[expert]).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_proj[expert])
            contribution = expert_output * scores[token_idx, slot_idx].unsqueeze(-1)
            result.index_add_(0, token_idx, contribution.to(result.dtype))
        return result

    def forward_dispatched(
        self, hidden: Tensor, tokens_per_expert: Tensor, scores: Tensor
    ) -> Tensor:
        """Apply contiguous local experts to MCore's expert-sorted token batches."""
        expert_count = self.gate_up_proj.shape[0]
        if tokens_per_expert.numel() != expert_count or scores.numel() != hidden.shape[0]:
            raise ValueError("Dispatched QwenAir expert tokens or scores have the wrong shape")
        counts = tokens_per_expert.tolist()
        if sum(counts) != hidden.shape[0]:
            raise ValueError("Dispatched token counts do not sum to input rows")
        pieces = []
        offset = 0
        for expert, count in enumerate(counts):
            source = hidden.narrow(0, offset, count)
            gate, up = F.linear(source, self.gate_up_proj[expert]).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_proj[expert])
            weighted = expert_output * scores.reshape(-1)[offset : offset + count].unsqueeze(-1)
            pieces.append(weighted.to(hidden.dtype))
            offset += count
        return torch.cat(pieces, dim=0)


class QwenAirTopKRouter(nn.Module):
    """FP32 global softmax, top-k, and selected-probability renormalization."""

    def __init__(self, config: QwenAirTextConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return raw logits, normalized top-k scores, and expert indices."""
        logits = F.linear(hidden, self.weight)
        probs = F.softmax(logits.float(), dim=-1)
        scores, indices = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        return logits, scores.to(logits.dtype), indices


class QwenAirSparseMoeBlock(nn.Module):
    """QwenAir routed experts plus a sigmoid-gated shared expert."""

    def __init__(self, config: QwenAirTextConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.gate = QwenAirTopKRouter(config)
        self.experts = QwenAirExperts(config, layer_idx=layer_idx)
        self.shared_expert = QwenAirMLP(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Return hidden output and router logits for optional global loss."""
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits, scores, indices = self.gate(flat)
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        routed = self.experts(flat, indices, scores)
        return (shared + routed).reshape(shape), logits


def _qwenair_router_statistics(
    router_logits: tuple[Tensor, ...], num_experts: int, top_k: int, token_mask: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Accumulate Qwen's per-expert hard counts and differentiable soft mass."""
    if not router_logits:
        raise ValueError("At least one router layer is required")
    count = torch.zeros(num_experts, device=router_logits[0].device, dtype=torch.float32)
    probabilities = torch.zeros_like(count)
    total_rows = torch.zeros((), device=count.device, dtype=torch.float32)
    mask = token_mask.reshape(-1).float() if token_mask is not None else None
    for logits in router_logits:
        probs = F.softmax(logits.float(), dim=-1)
        selected = probs.topk(top_k, dim=-1).indices
        if mask is None:
            count.scatter_add_(0, selected.flatten(), torch.ones_like(selected.flatten(), dtype=torch.float32))
            probabilities = probabilities + probs.sum(dim=0)
            total_rows = total_rows + logits.shape[0]
        else:
            count.scatter_add_(0, selected.flatten(), mask.repeat_interleave(top_k))
            probabilities = probabilities + (probs * mask.unsqueeze(-1)).sum(dim=0)
            total_rows = total_rows + mask.sum()
    return count, probabilities, total_rows


def qwenair_global_router_loss(
    router_logits: tuple[Tensor, ...], num_experts: int, top_k: int, token_mask: Tensor | None = None
) -> Tensor:
    """Compute Qwen's auxiliary load balancing loss across all decoder layers."""
    count, probabilities, total_rows = _qwenair_router_statistics(
        router_logits, num_experts, top_k, token_mask
    )
    if total_rows.item() == 0:
        raise ValueError("Router auxiliary loss requires at least one valid token")
    return num_experts * ((count / total_rows) * (probabilities / total_rows)).sum()
