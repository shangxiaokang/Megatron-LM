# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir sparse-attention selection and dense correctness reference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import QwenAirTextConfig
from .layers import QwenAirRMSNorm, apply_qwenair_rope


@dataclass
class QwenAirQSASelection:
    """Per-token QSA choice, with an optional dense correctness mask."""

    token_mask: Tensor | None
    block_starts: Tensor
    tail_tokens: Tensor


class QwenAirQSAIndexer(nn.Module):
    """Select complete visible key blocks and retain each incomplete tail."""

    def __init__(self, config: QwenAirTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (self.index_n_heads + self.index_kv_heads) * self.index_head_dim, bias=False
        )
        self.q_layernorm = QwenAirRMSNorm(self.index_head_dim, config.rms_norm_eps)
        self.k_layernorm = QwenAirRMSNorm(self.index_head_dim, config.rms_norm_eps)

    def forward(self, hidden: Tensor, cos: Tensor, sin: Tensor, visible: Tensor | None) -> QwenAirQSASelection:
        """Return an exact per-query selection for prefill/training.

        ``block_starts`` has shape [B,S,K] and stores the first physical token
        of each chosen four-token block.  Selection is per token, because the
        frozen HF implementation recomputes top-k for every query token.  A
        block-level TE kernel may consume it only when its contract can
        represent these choices without changing the mask.
        """
        batch, length, _ = hidden.shape
        if visible is not None and (visible.shape != (batch, length, length) or visible.dtype != torch.bool):
            raise ValueError("visible must be a boolean [batch, seq, seq] causal mask")
        projected = self.index_qk_proj(hidden)
        q_width = self.index_n_heads * self.index_head_dim
        q, raw_keys = torch.split(projected, (q_width, self.index_kv_heads * self.index_head_dim), dim=-1)
        q = q.unflatten(-1, (self.index_n_heads, self.index_head_dim))
        raw_keys = raw_keys.unflatten(-1, (self.index_kv_heads, self.index_head_dim)).squeeze(2)
        q = apply_qwenair_rope(self.q_layernorm(q), cos, sin)

        if visible is None:
            return self._select_unpadded(q, raw_keys, cos, sin)

        max_blocks = min(self.block_topk, math.ceil(length / self.compress_ratio))
        block_starts = torch.full((batch, length, max_blocks), -1, dtype=torch.int32, device=hidden.device)
        tail_tokens = torch.full(
            (batch, length, self.compress_ratio - 1), -1, dtype=torch.int32, device=hidden.device
        )
        selected_mask = torch.zeros_like(visible)
        for batch_idx in range(batch):
            for query_idx in range(length):
                local_visible = torch.nonzero(visible[batch_idx, query_idx], as_tuple=False).flatten()
                complete = local_visible.numel() // self.compress_ratio
                if complete:
                    blocks = local_visible[: complete * self.compress_ratio].reshape(complete, self.compress_ratio)
                    pooled = raw_keys[batch_idx].index_select(0, blocks.flatten())
                    pooled = pooled.reshape(complete, self.compress_ratio, self.index_head_dim)
                    pooled = pooled.float().mean(dim=1).to(raw_keys.dtype)
                    keys = self.k_layernorm(pooled).unsqueeze(0).unsqueeze(2)
                    key_cos = cos[batch_idx].index_select(0, blocks[:, 0]).unsqueeze(0)
                    key_sin = sin[batch_idx].index_select(0, blocks[:, 0]).unsqueeze(0)
                    keys = apply_qwenair_rope(keys, key_cos, key_sin).squeeze(0).squeeze(1)
                    with torch.autocast(device_type=hidden.device.type, enabled=False):
                        scores = torch.relu(q[batch_idx, query_idx].float() @ keys.float().T)
                        scores = scores.sum(dim=0) / math.sqrt(self.index_head_dim)
                    chosen = scores.topk(min(self.block_topk, complete), dim=0).indices
                    chosen_blocks = blocks.index_select(0, chosen)
                    block_starts[batch_idx, query_idx, : chosen.numel()] = chosen_blocks[:, 0].int()
                    selected_mask[batch_idx, query_idx, chosen_blocks.flatten()] = True
                tail = local_visible[complete * self.compress_ratio :]
                if tail.numel():
                    tail_tokens[batch_idx, query_idx, : tail.numel()] = tail.int()
                    selected_mask[batch_idx, query_idx, tail] = True
        return QwenAirQSASelection(selected_mask, block_starts, tail_tokens)

    def _select_unpadded(self, q: Tensor, raw_keys: Tensor, cos: Tensor, sin: Tensor) -> QwenAirQSASelection:
        """Select blocks in bounded query chunks without a sequence-square mask.

        This is for unpacked, unpadded causal self-attention. Full block keys
        are pooled once per layer, then a small query chunk scores all completed
        blocks. The resulting index tensor is [B,S,K], so temporary score
        storage is bounded by the chunk size rather than S squared. Arithmetic
        remains quadratic in S; a production long-context indexer needs a more
        efficient candidate-generation kernel.
        """
        batch, length, _, _ = q.shape
        ratio = self.compress_ratio
        full_blocks = length // ratio
        max_blocks = min(self.block_topk, math.ceil(length / ratio))
        block_starts = torch.full((batch, length, max_blocks), -1, dtype=torch.int32, device=q.device)
        offsets = torch.arange(ratio - 1, device=q.device)
        positions = torch.arange(length, device=q.device)
        tail_lengths = (positions + 1) % ratio
        tails = (positions // ratio)[:, None] * ratio + offsets
        tails = torch.where(offsets[None, :] < tail_lengths[:, None], tails, -1)
        tail_tokens = tails.to(torch.int32).unsqueeze(0).expand(batch, -1, -1).contiguous()
        if full_blocks == 0:
            return QwenAirQSASelection(None, block_starts, tail_tokens)

        pooled = raw_keys[:, : full_blocks * ratio].reshape(
            batch, full_blocks, ratio, self.index_head_dim
        )
        pooled = pooled.float().mean(dim=2).to(raw_keys.dtype)
        keys = self.k_layernorm(pooled).unsqueeze(2)
        keys = apply_qwenair_rope(
            keys, cos[:, : full_blocks * ratio : ratio], sin[:, : full_blocks * ratio : ratio]
        ).squeeze(2)
        keys_float = keys.float()
        for start in range(0, length, 16):
            end = min(start + 16, length)
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.einsum("bchd,bnd->bchn", q[:, start:end].float(), keys_float)
                scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)
            local_start = 0
            while start + local_start < end:
                complete = (start + local_start + 1) // ratio
                group_end = min(end, (complete + 1) * ratio - 1)
                local_end = group_end - start
                if complete:
                    chosen_count = min(self.block_topk, complete)
                    # topk's tie behavior depends on its input length. Match
                    # the frozen per-query reference by slicing the completed
                    # prefix before topk, rather than masking future blocks.
                    chosen = scores[:, local_start:local_end, :complete].topk(chosen_count, dim=-1).indices
                    block_starts[:, start + local_start : group_end, :chosen_count] = (chosen * ratio).to(
                        torch.int32
                    )
                local_start = local_end
        return QwenAirQSASelection(None, block_starts, tail_tokens)


def qsa_dense_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selected_token_mask: Tensor,
    *,
    scale: float | None = None,
    dropout_p: float = 0.0,
) -> Tensor:
    """Run causal GQA against an exact QSA mask as a correctness reference.

    Inputs and output use [batch, seq, heads, head_dim].  The boolean mask is
    [batch, query_seq, key_seq], with True meaning selected and visible.
    """
    batch, query_length, num_query_heads, head_dim = query.shape
    key_length = key.shape[1]
    num_kv_heads = key.shape[2]
    if value.shape[:3] != (batch, key_length, num_kv_heads):
        raise ValueError("QSA key and value geometry does not match")
    if num_query_heads % num_kv_heads:
        raise ValueError("QSA query heads must divide by KV heads")
    if selected_token_mask.shape != (batch, query_length, key_length):
        raise ValueError("QSA selected mask has the wrong shape")
    replication = num_query_heads // num_kv_heads
    key = key.repeat_interleave(replication, dim=2)
    value = value.repeat_interleave(replication, dim=2)
    has_keys = selected_token_mask.any(dim=-1)
    safe_mask = selected_token_mask.clone()
    safe_mask[..., 0] |= ~has_keys
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
        scores = scores * (scale if scale is not None else 1 / math.sqrt(head_dim))
        scores = scores.masked_fill(~safe_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        if dropout_p:
            probs = F.dropout(probs, p=dropout_p, training=True)
        output = torch.einsum("bhqk,bkhd->bqhd", probs, value.float())
    return (output * has_keys[:, :, None, None]).to(query.dtype)


class QwenAirQSA(nn.Module):
    """Qwen4-Exp gated attention with an indexer and dense reference backend."""

    def __init__(self, config: QwenAirTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.backend = config.qsa_backend
        self.compress_ratio = config.indexer_compress_ratio
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = QwenAirRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = QwenAirRMSNorm(self.head_dim, config.rms_norm_eps)
        self.indexer = QwenAirQSAIndexer(config, layer_idx)

    def forward(self, hidden: Tensor, cos: Tensor, sin: Tensor, visible: Tensor | None) -> Tensor:
        """Compute selected full attention and its sigmoid output gate."""
        if (
            hidden.is_cuda
            and hidden.dtype == torch.float32
            and not torch.is_autocast_enabled("cuda")
            and torch.backends.cuda.matmul.allow_tf32
        ):
            raise NotImplementedError(
                "Strict FP32 QwenAir QSA reference requires "
                "torch.backends.cuda.matmul.allow_tf32=False throughout forward and backward"
            )
        batch, length, _ = hidden.shape
        selected = self.indexer(hidden, cos, sin, visible)
        q_with_gate = self.q_proj(hidden).reshape(batch, length, self.num_heads, 2 * self.head_dim)
        query, gate = q_with_gate.chunk(2, dim=-1)
        query = apply_qwenair_rope(self.q_norm(query), cos, sin)
        key = apply_qwenair_rope(
            self.k_norm(self.k_proj(hidden).reshape(batch, length, self.num_kv_heads, self.head_dim)), cos, sin
        )
        value = self.v_proj(hidden).reshape(batch, length, self.num_kv_heads, self.head_dim)
        if self.backend in ("te_reference", "te_indexed_sdpa", "te_triton"):
            if self.compress_ratio != 4:
                raise NotImplementedError("TE QSA requires four-token compression")
            if self.training and self.attention_dropout:
                raise NotImplementedError("TE QSA does not support attention dropout")
            if query.dtype not in (torch.bfloat16, torch.float32):
                raise NotImplementedError("TE QSA requires BF16 or FP32 inputs")
            if visible is not None:
                full_causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
                if not torch.equal(visible, full_causal.unsqueeze(0).expand(batch, -1, -1)):
                    raise NotImplementedError("TE QSA requires unpacked, unpadded causal sequences")
            valid = selected.block_starts >= 0
            if torch.any(valid & (selected.block_starts % 4 != 0)):
                raise NotImplementedError("TE QSA requires physical four-token block alignment")
            selected_blocks = torch.where(valid, selected.block_starts // 4, -1)
            try:
                if self.backend == "te_triton":
                    from transformer_engine.pytorch import (
                        qsa_triton_attention as attention_op,
                    )
                elif self.backend == "te_indexed_sdpa":
                    from transformer_engine.pytorch import (
                        qsa_indexed_sdpa_attention as attention_op,
                    )
                else:
                    from transformer_engine.pytorch import (
                        qsa_block_sparse_attention as attention_op,
                    )
            except ImportError as error:
                raise ImportError(
                    f"Install the QwenAir Transformer Engine attention API for qsa_backend={self.backend!r}"
                ) from error
            if value.dtype != query.dtype:
                # With FP32 weights under BF16 autocast, RoPE's FP32 cos/sin
                # promote Q/K to FP32 while the unrotated V stays BF16. BF16
                # values convert exactly to FP32, preserving dense-reference
                # accumulation and TE's same-dtype Q/K/V contract.
                if query.dtype == key.dtype == torch.float32 and value.dtype == torch.bfloat16:
                    value = value.float()
                else:
                    raise TypeError(
                        "TE QSA requires matching Q/K/V dtypes after exact BF16 V promotion"
                    )
            attention_kwargs = {"scale": self.head_dim**-0.5}
            if self.backend == "te_triton":
                # The indices are produced by the local causal selector above.
                # Skip TE's synchronizing sort/range check on every layer.
                attention_kwargs["validate_indices"] = False
            output = attention_op(query, key, value, selected_blocks, **attention_kwargs)
        else:
            if selected.token_mask is None:
                raise ValueError("Dense QSA requires an explicit selected token mask")
            output = qsa_dense_attention(
                query, key, value, selected.token_mask,
                scale=self.head_dim**-0.5,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
        return self.o_proj((output * torch.sigmoid(gate)).flatten(-2))
