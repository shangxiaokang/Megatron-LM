# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir single-rank text model and causal language-model training loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    QwenAirTopKRouter,
    inject_hyper_output,
    qwenair_global_router_loss,
    qwenair_rope,
)
from .ple import QwenAirNGramEmbedding, QwenAirPLE, qwenair_ngram_metadata
from .qsa import QwenAirQSA

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


@dataclass
class QwenAirOutput:
    """Text training result with optional cross-layer router loss."""

    logits: Tensor
    loss: Tensor | None = None
    aux_loss: Tensor | None = None
    router_logits: tuple[Tensor, ...] | None = None


def _validate_te_qsa_token_mask(token_mask: Tensor) -> None:
    """Reject masks whose visible tokens are not a nonempty prefix.

    TE QSA has no arbitrary attention-mask input. Contiguous right padding is
    nevertheless safe: causal queries in the valid prefix cannot attend to
    later padding, while the loss, GDN, PLE, and router auxiliary objective
    continue to use ``token_mask`` to ignore that padded suffix.
    """
    if not bool(torch.all(token_mask.any(dim=-1))):
        raise NotImplementedError("TE QSA requires at least one valid token in every sequence")
    if token_mask.shape[1] > 1 and bool(torch.any(token_mask[:, 1:] & ~token_mask[:, :-1])):
        raise NotImplementedError(
            "TE QSA supports only unpadded sequences or a contiguous right-padded suffix"
        )


class QwenAirDecoderLayer(nn.Module):
    """A QwenAir GDN/QSA layer with MoE and two gated residual cells."""

    def __init__(
        self,
        config: QwenAirTextConfig,
        layer_idx: int,
        ple_process_group: dist.ProcessGroup | None = None,
        ep_group: dist.ProcessGroup | None = None,
        expert_tp_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = QwenAirGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = QwenAirQSA(config, layer_idx)
        if ep_group is None:
            self.mlp = QwenAirSparseMoeBlock(config, layer_idx)
        else:
            from .moe_ep import QwenAirExpertParallelBlock

            self.mlp = QwenAirExpertParallelBlock(
                config, ep_group, expert_tp_group, layer_idx
            )
        ple_layer_index = (
            config.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in config.ple_layer_ids
            else None
        )
        self.ple = (
            QwenAirPLE(config, layer_idx, ple_layer_index, ple_process_group)
            if ple_layer_index is not None
            else None
        )
        self.attn_hyper_connection = QwenAirGatedResidual(config)
        self.mlp_hyper_connection = QwenAirGatedResidual(config)

    def forward(
        self,
        hidden: Tensor,
        input_ids: Tensor | None,
        cos: Tensor,
        sin: Tensor,
        visible: Tensor | None,
        token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply PLE, attention, experts, and both stream injections."""
        if self.ple is not None:
            if input_ids is None:
                raise ValueError("PLE requires the original token IDs")
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
        self,
        config: QwenAirTextConfig,
        ple_process_group: dist.ProcessGroup | None = None,
        ep_group: dist.ProcessGroup | None = None,
        expert_tp_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            QwenAirDecoderLayer(config, idx, ple_process_group, ep_group, expert_tp_group)
            for idx in range(config.num_hidden_layers)
        )
        self.hyper_connection_mixer = QwenAirGatedResidual(config, use_combine=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        ple_input_ids: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return hidden states; embeddings may contain scattered visual features.

        ``ple_input_ids`` carries the original token stream used for n-gram
        hashing when visual embeddings replace token embeddings at some slots.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            if ple_input_ids is not None:
                raise ValueError("ple_input_ids is only used with inputs_embeds")
            batch, length = input_ids.shape
            embeddings = self.embed_tokens(input_ids)
            original_ids = input_ids
        else:
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
                raise ValueError("inputs_embeds must have shape [batch, sequence, hidden_size]")
            batch, length, _ = inputs_embeds.shape
            embeddings = inputs_embeds
            original_ids = ple_input_ids
            if original_ids is not None and original_ids.shape != (batch, length):
                raise ValueError("ple_input_ids must have shape [batch, sequence]")
            if self.config.ple_layer_ids and original_ids is None:
                raise ValueError("PLE requires ple_input_ids with inputs_embeds")
        if length < 1:
            raise ValueError("QwenAir requires at least one token")
        if (
            self.config.qsa_backend == "dense"
            and length > self.config.max_reference_sequence_length
        ):
            raise NotImplementedError(
                "QwenAir dense selector/reference is limited to "
                f"{self.config.max_reference_sequence_length} tokens; long-context training needs a sparse backend"
            )
        if attention_mask is None:
            token_mask = torch.ones((batch, length), device=embeddings.device, dtype=torch.bool)
        else:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            token_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(length, device=embeddings.device).expand(batch, -1)
        cos, sin = qwenair_rope(self.config, position_ids, self.embed_tokens.weight.dtype)
        if cos.shape[:2] != (batch, length):
            raise ValueError("position_ids must match input_ids")
        if self.config.qsa_backend in ("te_reference", "te_indexed_sdpa", "te_triton"):
            if attention_mask is not None:
                _validate_te_qsa_token_mask(token_mask)
            visible = None
        else:
            causal = torch.ones(length, length, device=embeddings.device, dtype=torch.bool).tril()
            visible = causal.unsqueeze(0) & token_mask[:, :, None] & token_mask[:, None, :]
        hidden = embeddings.repeat(1, 1, self.config.hc_count)
        ids_for_ple = (
            torch.where(token_mask, original_ids, self.config.eos_token_id)
            if self.config.ple_layer_ids
            else original_ids
        )
        router_logits = []
        for layer in self.layers:
            hidden, logits = layer(hidden, ids_for_ple, cos, sin, visible, token_mask)
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
        self,
        config: QwenAirTextConfig,
        ple_process_group: dist.ProcessGroup | None = None,
        ep_group: dist.ProcessGroup | None = None,
        expert_tp_group: dist.ProcessGroup | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config)
        if pg_collection is not None:
            if not hasattr(pg_collection, "ep") or not hasattr(pg_collection, "expt_tp"):
                raise ValueError("QwenAir pg_collection requires ep and expt_tp groups")
            if ep_group is not None and ep_group is not pg_collection.ep:
                raise ValueError("QwenAir received conflicting ep_group and pg_collection.ep")
            if expert_tp_group is not None and expert_tp_group is not pg_collection.expt_tp:
                raise ValueError(
                    "QwenAir received conflicting expert_tp_group and pg_collection.expt_tp"
                )
            if ple_process_group is not None and ple_process_group is not pg_collection.ep:
                raise ValueError(
                    "QwenAir received conflicting ple_process_group and pg_collection.ep"
                )
            ep_group = pg_collection.ep
            expert_tp_group = pg_collection.expt_tp
            if config.ple_layer_ids:
                ple_process_group = pg_collection.ep
        if (ep_group is None) != (expert_tp_group is None):
            raise ValueError("QwenAir EP requires both ep_group and expert_tp_group")
        if config.expert_model_parallel_size > 1 and ep_group is None:
            raise ValueError("QwenAir expert_model_parallel_size > 1 requires an explicit ep_group")
        if ep_group is not None:
            ep_size = dist.get_world_size(ep_group)
            if ep_size != config.expert_model_parallel_size:
                raise ValueError(
                    "QwenAir ep_group size must match config.expert_model_parallel_size"
                )
            if dist.get_world_size(expert_tp_group) != config.expert_tensor_parallel_size:
                raise ValueError(
                    "QwenAir expert_tp_group size must match config.expert_tensor_parallel_size"
                )
        if config.ple_layer_ids and ep_group is not None and ple_process_group is not ep_group:
            raise NotImplementedError(
                "QwenAir PLE and expert sharding currently require the same process group"
            )
        self.ple_process_group = ple_process_group
        self.ep_group = ep_group
        self.pg_collection = pg_collection
        self._check_single_rank_resources(config, ple_process_group, ep_group)
        self.model = QwenAirTextModel(config, ple_process_group, ep_group, expert_tp_group)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._mark_sharded_parameters_for_ddp()
        self._initialize_weights(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _mark_sharded_parameters_for_ddp(self) -> None:
        """Route PLE row shards through MCore's expert-DP gradient bucket.

        MCore DDP uses ``parameter.allreduce=False`` for model-parallel shards
        whose replicas reduce over ``expt_dp`` instead of the dense ``dp``
        group.  Routed expert weights stamp this attribute in their owning
        module; PLE row shards need the same ownership because they share the
        QwenAir EP group.
        """
        for module in self.modules():
            if isinstance(module, QwenAirNGramEmbedding) and module.group_size > 1:
                setattr(module.ngram_embedding.weight, "allreduce", False)

    @staticmethod
    def _check_single_rank_resources(
        config: QwenAirTextConfig,
        ple_process_group: dist.ProcessGroup | None = None,
        ep_group: dist.ProcessGroup | None = None,
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
            config.num_hidden_layers
            * config.num_experts
            * 3
            * config.moe_intermediate_size
            * config.hidden_size
        )
        if ep_group is not None:
            if not dist.is_initialized() or config.num_experts % dist.get_world_size(ep_group):
                raise ValueError(
                    "QwenAir EP requires an initialized group dividing the expert count"
                )
            expert_parameters //= dist.get_world_size(ep_group)
        if expert_parameters > config.max_single_rank_parameters:
            raise ValueError(
                "QwenAir experts need distributed sharding before target-size construction"
            )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Mark PLE rows and local expert rows as axis-0 MCore shards."""
        if self.ple_process_group is None and self.ep_group is None:
            return super().sharded_state_dict(prefix, sharded_offsets, metadata)
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        from .moe_ep import QwenAirExpertParallelBlock

        axis_map = {
            f"{name}.ngram_embedding.weight": 0
            for name, module in self.named_modules()
            if isinstance(module, QwenAirNGramEmbedding) and module.group_size > 1
        }
        for name, module in self.named_modules():
            if isinstance(module, QwenAirExpertParallelBlock):
                axis_map[f"{name}.experts.gate_up_proj"] = 0
                axis_map[f"{name}.experts.down_proj"] = 0
        # PLE/expert offsets vary across EP while replicas of each offset vary
        # across EDP. Dense tensors are replicas across both axes. Passing
        # WORLD here would assign every EP offset a distinct DP replica id and
        # could cause DCP to omit nonzero EP shards as non-main replicas.
        checkpoint_dp_group = (
            self.pg_collection.expt_dp
            if self.pg_collection is not None and hasattr(self.pg_collection, "expt_dp")
            else (metadata or {}).get("dp_cp_group")
        )
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            axis_map,
            sharded_offsets,
            tp_group=self.ep_group if self.ep_group is not None else self.ple_process_group,
            dp_cp_group=checkpoint_dp_group,
        )

    def sync_ep_replicated_gradients(self) -> None:
        """SUM replicated EP gradients once when not using MCore DDP.

        Call only after backward in a standalone EP training loop. MCore DDP
        must own this synchronization instead. Local PLE and expert shards are
        deliberately excluded, and globally unused parameters keep grad=None.
        """
        if self.ep_group is None:
            return
        from .moe_ep import QwenAirExpertParallelBlock

        local_shards = {
            f"{name}.ngram_embedding.weight"
            for name, module in self.named_modules()
            if isinstance(module, QwenAirNGramEmbedding) and module.group_size > 1
        }
        for name, module in self.named_modules():
            if isinstance(module, QwenAirExpertParallelBlock):
                local_shards.update((f"{name}.experts.gate_up_proj", f"{name}.experts.down_proj"))
        for name, parameter in self.named_parameters():
            if name in local_shards:
                continue
            has_grad = torch.tensor(
                int(parameter.grad is not None), device=parameter.device, dtype=torch.int32
            )
            dist.all_reduce(has_grad, group=self.ep_group)
            if has_grad.item() == 0:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, group=self.ep_group)

    def _initialize_weights(self, config: QwenAirTextConfig) -> None:
        """Match HF zero-centered norms, linear initialization, and PLE zero conv."""
        ple_embeddings = {
            module.ngram_embedding
            for module in self.modules()
            if isinstance(module, QwenAirNGramEmbedding)
        }
        ple_convolutions = {
            module.conv1d for module in self.modules() if isinstance(module, QwenAirPLE)
        }
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                if module in ple_convolutions:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in ple_embeddings:
                nn.init.normal_(module.weight, std=config.initializer_range)
            elif isinstance(module, QwenAirTopKRouter):
                nn.init.normal_(module.weight, std=config.initializer_range)

    def set_input_tensor(self, input_tensor: Tensor | list[Tensor | None] | None) -> None:
        """Accept MCore's empty PP input while pipeline parallelism is disabled.

        The forward/backward schedule calls this hook even for ``PP=1``. A
        nonempty pipeline input would imply an unsupported PP layout, so it is
        rejected instead of being silently ignored.
        """
        if input_tensor is None:
            return
        if isinstance(input_tensor, (list, tuple)) and len(input_tensor) == 1:
            if input_tensor[0] is None:
                return
        raise NotImplementedError("QwenAir set_input_tensor supports only the PP=1 empty input")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_router_logits: bool = False,
        enable_mtp: bool = False,
        labels_are_shifted: bool = False,
        *,
        inputs_embeds: Tensor | None = None,
        ple_input_ids: Tensor | None = None,
    ) -> QwenAirOutput:
        """Compute next-token CE and optional Qwen cross-layer MoE auxiliary loss."""
        if enable_mtp:
            raise NotImplementedError("QwenAir MTP training shift/loss contract is unavailable")
        hidden, router_logits = self.model(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError("labels must match the input batch and sequence")
            if labels_are_shifted:
                shifted_logits = logits.float().contiguous()
                shifted_labels = labels.contiguous()
            else:
                shifted_logits = logits[:, :-1].float().contiguous()
                shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum" if self.ep_group is not None else "mean",
            )
            if self.ep_group is not None:
                valid_labels = (shifted_labels != -100).sum().to(dtype=torch.float32)
                dist.all_reduce(valid_labels, group=self.ep_group)
                if valid_labels.item() == 0:
                    raise ValueError("QwenAir EP loss requires at least one valid label")
                loss = loss / valid_labels
        aux_loss = None
        if output_router_logits:
            if self.ep_group is None:
                aux_loss = qwenair_global_router_loss(
                    router_logits,
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    attention_mask,
                )
            else:
                from .moe_ep import qwenair_ep_router_loss

                aux_loss = qwenair_ep_router_loss(
                    router_logits,
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    self.ep_group,
                    attention_mask,
                )
            if loss is not None:
                loss = loss + self.config.router_aux_loss_coef * aux_loss
        return QwenAirOutput(
            logits=logits,
            loss=loss,
            aux_loss=aux_loss,
            router_logits=router_logits if output_router_logits else None,
        )


QwenAirModel = QwenAirForCausalLM
