# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir single-rank text model and causal language-model training loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from megatron.core.transformer.module import MegatronModule

from .config import QwenAirTextConfig
from .layers import (
    QwenAirGatedDeltaNet,
    QwenAirGatedResidual,
    QwenAirSparseMoeBlock,
    inject_hyper_output,
    qwenair_global_router_loss,
    qwenair_rope,
)
from .ple import QwenAirNGramEmbedding, QwenAirPLE, qwenair_ngram_metadata
from .qsa import QwenAirQSA


@dataclass
class QwenAirOutput:
    """Text training result with optional cross-layer router loss."""

    logits: Tensor
    loss: Tensor | None = None
    aux_loss: Tensor | None = None
    router_logits: tuple[Tensor, ...] | None = None


class QwenAirDecoderLayer(nn.Module):
    """A QwenAir GDN/QSA layer with MoE and two gated residual cells."""

    def __init__(
        self,
        config: QwenAirTextConfig,
        layer_idx: int,
        ple_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = QwenAirGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = QwenAirQSA(config, layer_idx)
        self.mlp = QwenAirSparseMoeBlock(config)
        ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = (
            QwenAirPLE(config, layer_idx, ple_layer_index, ple_process_group)
            if ple_layer_index is not None else None
        )
        self.attn_hyper_connection = QwenAirGatedResidual(config)
        self.mlp_hyper_connection = QwenAirGatedResidual(config)

    def forward(
        self,
        hidden: Tensor,
        input_ids: Tensor,
        cos: Tensor,
        sin: Tensor,
        visible: Tensor,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply PLE, attention, experts, and both stream injections."""
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, input_ids, token_mask)
        read, original, injection = self.attn_hyper_connection(hidden)
        if self.layer_type == "linear_attention":
            attention = self.linear_attn(read, token_mask)
        else:
            attention = self.self_attn(read, cos, sin, visible)
        hidden = inject_hyper_output(original, attention, injection)
        read, original, injection = self.mlp_hyper_connection(hidden)
        expert_output, router_logits = self.mlp(read)
        return inject_hyper_output(original, expert_output, injection), router_logits


class QwenAirTextModel(nn.Module):
    """QwenAir text decoder with an HF-compatible logical state dict."""

    def __init__(
        self, config: QwenAirTextConfig, ple_process_group: dist.ProcessGroup | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            QwenAirDecoderLayer(config, idx, ple_process_group)
            for idx in range(config.num_hidden_layers)
        )
        self.hyper_connection_mixer = QwenAirGatedResidual(config, use_combine=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return contracted hidden states and per-layer raw router logits."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length < 1:
            raise ValueError("QwenAir requires at least one token")
        if length > self.config.max_reference_sequence_length:
            raise NotImplementedError(
                "QwenAir dense selector/reference is limited to "
                f"{self.config.max_reference_sequence_length} tokens; long-context training needs a sparse backend"
            )
        if attention_mask is None:
            token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            token_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device).expand(batch, -1)
        cos, sin = qwenair_rope(self.config, position_ids, self.embed_tokens.weight.dtype)
        if cos.shape[:2] != (batch, length):
            raise ValueError("position_ids must match input_ids")
        causal = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).tril()
        visible = causal.unsqueeze(0) & token_mask[:, :, None] & token_mask[:, None, :]
        hidden = self.embed_tokens(input_ids).repeat(1, 1, self.config.hc_count)
        ple_input_ids = (
            torch.where(token_mask, input_ids, self.config.eos_token_id)
            if self.config.ple_layer_ids else input_ids
        )
        router_logits = []
        for layer in self.layers:
            hidden, logits = layer(hidden, ple_input_ids, cos, sin, visible, token_mask)
            router_logits.append(logits)
        contracted = self.hyper_connection_mixer(hidden)
        return contracted, tuple(router_logits)


class QwenAirForCausalLM(MegatronModule):
    """Trainable QwenAir text reference with explicit MTP/indexer boundaries.

    The forward is a small-context reference; PLE can use row-sharded
    checkpoints on an explicit process group. The indexer uses a hard
    top-k mask, so LM loss cannot train its projection.  An independent indexer
    target/loss is required for exact QwenAir pretraining and is not invented
    here.  MTP is also excluded pending its authoritative training contract.
    """

    def __init__(
        self, config: QwenAirTextConfig, ple_process_group: dist.ProcessGroup | None = None
    ) -> None:
        super().__init__(config)
        self.ple_process_group = ple_process_group
        self._check_single_rank_resources(config, ple_process_group)
        self.model = QwenAirTextModel(config, ple_process_group)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._initialize_weights(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _check_single_rank_resources(
        config: QwenAirTextConfig, ple_process_group: dist.ProcessGroup | None = None
    ) -> None:
        """Refuse target-size allocation before materializing any giant tensor."""
        if config.ple_layer_ids:
            if ple_process_group is not None and not dist.is_initialized():
                raise ValueError("PLE process_group requires initialized torch.distributed")
            group_size = (
                dist.get_world_size(ple_process_group) if ple_process_group is not None else 1
            )
            heads = (config.ngram_size - 1) * config.heads_per_ngram
            head_width = config.ple_embed_dim // heads
            for ple_index in range(len(config.ple_layer_ids)):
                _, _, _, table_rows = qwenair_ngram_metadata(config, ple_index)
                if (
                    table_rows % group_size
                    or table_rows // group_size * head_width > config.max_single_rank_ple_elements
                ):
                    raise ValueError(
                        "QwenAir PLE needs a distributed table before target-size construction"
                    )
        expert_parameters = (
            config.num_hidden_layers * config.num_experts * 3
            * config.moe_intermediate_size * config.hidden_size
        )
        if expert_parameters > config.max_single_rank_parameters:
            raise ValueError("QwenAir experts need distributed sharding before target-size construction")

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Mark only PLE table rows as axis-0 shards in MCore checkpoints."""
        if self.ple_process_group is None:
            return super().sharded_state_dict(prefix, sharded_offsets, metadata)
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        axis_map = {
            f"{name}.ngram_embedding.weight": 0
            for name, module in self.named_modules()
            if isinstance(module, QwenAirNGramEmbedding) and module.group_size > 1
        }
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            axis_map,
            sharded_offsets,
            tp_group=self.ple_process_group,
            dp_cp_group=(metadata or {}).get("dp_cp_group"),
        )

    def _initialize_weights(self, config: QwenAirTextConfig) -> None:
        """Match HF zero-centered norms, linear initialization, and PLE zero conv."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=config.initializer_range)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_router_logits: bool = False,
        enable_mtp: bool = False,
    ) -> QwenAirOutput:
        """Compute next-token CE and optional Qwen cross-layer MoE auxiliary loss."""
        if enable_mtp:
            raise NotImplementedError("QwenAir MTP training shift/loss contract is unavailable")
        hidden, router_logits = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            shifted_logits = logits[:, :-1].float().contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1), ignore_index=-100,
            )
        aux_loss = None
        if output_router_logits:
            aux_loss = qwenair_global_router_loss(
                router_logits, self.config.num_experts, self.config.num_experts_per_tok, attention_mask
            )
            if loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss
        return QwenAirOutput(
            logits=logits, loss=loss, aux_loss=aux_loss,
            router_logits=router_logits if output_router_logits else None,
        )


QwenAirModel = QwenAirForCausalLM
