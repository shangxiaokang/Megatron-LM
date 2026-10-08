# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir per-layer hashed n-gram embedding for training sequences."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import Tensor, nn
from torch.nn import functional as F

from .config import QwenAirTextConfig
from .initialization import initialize_qwenair_sharded_normal_
from .layers import QwenAirRMSNorm

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(vocab_size: int, ngram_size: int, layer_index: int, seed: int) -> Tensor:
    multiplier_max = ((1 << 63) - 1) // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + 10007 * layer_index
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def qwenair_ngram_metadata(
    config: QwenAirTextConfig, ple_layer_index: int = 0
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Compute the logical PLE table layout without allocating embedding rows."""
    head_count = (config.ngram_size - 1) * config.heads_per_ngram
    sizes = [
        _find_nth_prime_after(config.ngram_vocab_size_base - 1, ple_layer_index * head_count + head + 1)
        for head in range(head_count)
    ]
    offsets = []
    total_rows = 0
    for size in sizes:
        offsets.append(total_rows)
        total_rows += size
    divisor = config.make_ngram_vocab_size_divisible_by
    padded_rows = math.ceil(total_rows / divisor) * divisor
    return (
        _build_layer_multipliers(config.vocab_size, config.ngram_size, ple_layer_index, config.seed),
        torch.tensor(sizes, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        padded_rows,
    )


def _shift_right_ignore_eos(input_ids: Tensor, shift: int, eos_token_id: int) -> Tensor:
    if not shift:
        return input_ids
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    eos_positions = torch.where(input_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat((eos_positions.new_full((batch, 1), -1), previous_eos_inclusive[:, :-1]), dim=1)
    source_positions = positions - shift
    shifted = input_ids.gather(1, source_positions.clamp_min(0).expand(batch, -1))
    valid = (positions.unsqueeze(0) - previous_eos - 1 >= shift) & (source_positions >= 0)
    return torch.where(valid, shifted, eos_token_id)


def _hash_ngram_indices(
    input_ids: Tensor,
    eos_token_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    multipliers: Tensor,
    vocab_sizes: Tensor,
    offsets: Tensor,
) -> Tensor:
    input_ids = input_ids.long()
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("PLE input_ids must be nonempty [batch, sequence]")
    context = input_ids.new_full((input_ids.shape[0], ngram_size - 1), eos_token_id)
    history = torch.cat((context, input_ids), dim=1)
    shifted = [_shift_right_ignore_eos(history, amount, eos_token_id) for amount in range(ngram_size)]
    blocks = []
    for order in range(2, ngram_size + 1):
        start = (order - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, order):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        blocks.append(torch.remainder(mixed.unsqueeze(-1), vocab_sizes[start:end]) + offsets[start:end])
    return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]


def qwenair_ngram_indices(
    config: QwenAirTextConfig, input_ids: Tensor, ple_layer_index: int = 0
) -> Tensor:
    """Hash a PLE input sequence without constructing the giant target table."""
    multipliers, sizes, offsets, _ = qwenair_ngram_metadata(config, ple_layer_index)
    eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    return _hash_ngram_indices(
        input_ids, eos_token_id, config.ngram_size, config.heads_per_ngram,
        multipliers.to(input_ids.device), sizes.to(input_ids.device), offsets.to(input_ids.device),
    )


class QwenAirNGramEmbedding(nn.Module):
    """EOS-aware hash table with optional row shards and variable-length lookup.

    The local ``ngram_embedding.weight`` is a contiguous row slice of the
    padded logical HF table. An explicit process group may be the MCore TP
    group or a dedicated table group; no global parallel-state lookup occurs.
    """

    def __init__(
        self,
        config: QwenAirTextConfig,
        layer_idx: int,
        ple_layer_index: int,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if process_group is not None and not dist.is_initialized():
            raise ValueError(
                "PLE process_group requires an initialized torch.distributed process group"
            )
        self.process_group = process_group
        self.group_size = dist.get_world_size(process_group) if process_group is not None else 1
        self.group_rank = dist.get_rank(process_group) if process_group is not None else 0
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        head_width = config.ple_embed_dim // self.ngram_heads
        multipliers, sizes, offsets, padded_rows = qwenair_ngram_metadata(config, ple_layer_index)
        if padded_rows % self.group_size:
            raise ValueError("Padded PLE rows must divide evenly across the table process group")
        self.padded_rows = padded_rows
        self.rows_per_rank = padded_rows // self.group_size
        self.shard_start = self.group_rank * self.rows_per_rank
        self.shard_end = self.shard_start + self.rows_per_rank
        if self.rows_per_rank * head_width > config.max_single_rank_ple_elements:
            raise ValueError(
                f"PLE needs {self.rows_per_rank * head_width} elements on this rank; "
                "increase table shards or the explicit resource limit"
            )
        self.register_buffer("layer_multipliers", multipliers)
        self.register_buffer("ngram_heads_vocab_sizes", sizes)
        self.register_buffer("ngram_heads_offsets", offsets)
        weight = torch.empty(self.rows_per_rank, head_width)
        self.ngram_embedding = nn.Embedding(
            self.rows_per_rank, head_width, _weight=weight
        )
        initialize_qwenair_sharded_normal_(
            self.ngram_embedding.weight,
            global_element_start=self.shard_start * head_width,
            logical_numel=self.padded_rows * head_width,
            base_seed=config.seed,
            namespace="qwenair.ple.table",
            layer_idx=layer_idx,
            std=config.initializer_range,
        )

    def lookup_indices(self, indices: Tensor) -> Tensor:
        """Route arbitrary local index counts to row owners and return their values.

        The float-valued return exchange uses PyTorch's differentiable all-to-all
        operator. Its backward sends each requester's gradients to the owning
        shard, where ``F.embedding`` accumulates the correct local row gradient.
        """
        weight = self.ngram_embedding.weight
        if indices.device != weight.device:
            raise ValueError("PLE indices and table shard must be on the same device")
        flat = indices.long().reshape(-1)
        if torch.any((flat < 0) | (flat >= self.padded_rows)):
            raise ValueError("PLE lookup index is outside the padded logical table")
        if self.group_size == 1:
            return F.embedding(indices.long(), weight)

        owners = torch.div(flat, self.rows_per_rank, rounding_mode="floor")
        order = owners.argsort(stable=True)
        send_ids = flat.index_select(0, order).contiguous()
        send_counts = torch.bincount(owners, minlength=self.group_size).to(torch.int64)
        receive_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(receive_counts, send_counts, group=self.process_group)
        send_splits = send_counts.tolist()
        receive_splits = receive_counts.tolist()

        receive_ids = torch.empty(sum(receive_splits), dtype=torch.long, device=flat.device)
        dist.all_to_all_single(
            receive_ids, send_ids,
            output_split_sizes=receive_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )
        local_ids = receive_ids - self.shard_start
        local_values = F.embedding(local_ids, weight)
        returned = torch.empty(
            flat.numel(), weight.shape[1], dtype=weight.dtype, device=weight.device
        )
        returned = dist_nn.all_to_all_single(
            returned, local_values,
            output_split_sizes=send_splits,
            input_split_sizes=receive_splits,
            group=self.process_group,
        )
        inverse_order = order.argsort()
        return returned.index_select(0, inverse_order).reshape(*indices.shape, weight.shape[1])

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Describe the logical HF table as equal axis-0 checkpoint shards."""
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        state = self.state_dict(prefix="", keep_vars=True)
        axis_map = {"ngram_embedding.weight": 0} if self.group_size > 1 else {}
        dp_cp_group = (metadata or {}).get("dp_cp_group")
        return make_sharded_tensors_for_checkpoint(
            state, prefix, axis_map, sharded_offsets,
            tp_group=self.process_group, dp_cp_group=dp_cp_group,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Hash n-grams within each EOS-delimited segment and look up features."""
        indices = _hash_ngram_indices(
            input_ids, self.eos_token_id, self.ngram_size, self.heads_per_ngram,
            self.layer_multipliers, self.ngram_heads_vocab_sizes, self.ngram_heads_offsets,
        )
        return self.lookup_indices(indices).flatten(-2)


class QwenAirPLE(nn.Module):
    """Hash, gate, and dilated depthwise convolution before a GDN layer."""

    def __init__(
        self,
        config: QwenAirTextConfig,
        layer_idx: int,
        ple_layer_index: int,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        width = self.hc_count * self.hidden_size
        self.ple_embedding = QwenAirNGramEmbedding(
            config, layer_idx, ple_layer_index, process_group
        )
        self.key_proj = nn.Linear(config.ple_embed_dim, width, bias=False)
        self.value_proj = nn.Linear(config.ple_embed_dim, config.hidden_size, bias=False)
        self.norm_key = QwenAirRMSNorm(width, config.rms_norm_eps, group_size=config.hidden_size)
        self.norm_query = QwenAirRMSNorm(width, config.rms_norm_eps, group_size=config.hidden_size)
        self.norm_conv = QwenAirRMSNorm(width, config.rms_norm_eps, group_size=config.hidden_size)
        self.conv1d = nn.Conv1d(
            width, width, config.ple_conv_kernel_size, groups=width, dilation=config.ngram_size, bias=False
        )
        self.short_conv_state_len = (config.ple_conv_kernel_size - 1) * config.ngram_size
        nn.init.zeros_(self.conv1d.weight)

    def forward(self, hidden: Tensor, input_ids: Tensor, token_mask: Tensor | None = None) -> Tensor:
        """Return the additive four-stream PLE update for a training sequence."""
        features = self.ple_embedding(input_ids)
        key = self.norm_key(self.key_proj(features)).unflatten(-1, (self.hc_count, self.hidden_size))
        query = self.norm_query(hidden).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(features)
        score = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        transformed = torch.sqrt(score.abs().clamp_min(1e-6)) * score.sign()
        gated = (torch.sigmoid(transformed) * value.unsqueeze(-2)).flatten(-2)
        normalized = self.norm_conv(gated)
        if token_mask is not None:
            gated = gated * token_mask.unsqueeze(-1)
            normalized = normalized * token_mask.unsqueeze(-1)
        conv_input = F.pad(normalized.transpose(1, 2), (self.short_conv_state_len, 0))
        conv_output = F.silu(self.conv1d(conv_input)).transpose(1, 2)
        return gated + conv_output
