# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small QwenAir expert-parallel path using MCore's all-to-all dispatcher.

The text model accepts explicit expert-parallel groups and keeps only the local
packed experts on each rank, while retaining the frozen HF parameter names.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import Tensor, nn
from torch.nn import functional as F

from .config import QwenAirTextConfig
from .layers import QwenAirExperts, QwenAirMLP, QwenAirTopKRouter, _qwenair_router_statistics


def qwenair_ep_router_loss(
    router_logits: tuple[Tensor, ...],
    num_experts: int,
    top_k: int,
    ep_group: dist.ProcessGroup,
    token_mask: Tensor | None = None,
) -> Tensor:
    """Return Qwen's global load loss with one dense-global gradient across EP.

    Every EP rank sees the same forward value. The straight-through gradient
    factor compensates for the backward all-reduce of the differentiable
    probability sum when all EP ranks backpropagate their local main loss plus
    this global auxiliary value. Replicated router gradients must subsequently
    be summed once across EP, either by a trainer or by the prototype's explicit
    ``sync_replicated_gradients`` method.
    """
    if not dist.is_initialized():
        raise ValueError("QwenAir EP router loss requires initialized torch.distributed")
    count, probability_sum, total_rows = _qwenair_router_statistics(
        router_logits, num_experts, top_k, token_mask
    )
    dist.all_reduce(count, group=ep_group)
    probability_sum = dist_nn.all_reduce(probability_sum, group=ep_group)
    dist.all_reduce(total_rows, group=ep_group)
    if total_rows.item() == 0:
        raise ValueError("Router auxiliary loss requires at least one valid token")
    global_loss = num_experts * (
        (count / total_rows) * (probability_sum / total_rows)
    ).sum()
    group_size = dist.get_world_size(ep_group)
    return global_loss.detach() + (global_loss - global_loss.detach()) / group_size


class QwenAirExpertParallelBlock(nn.Module):
    """QwenAir top-k MoE with local packed experts and MCore EP communication.

    This prototype supports expert parallelism with expert TP size one on CUDA.
    Router and shared-expert parameters are replicated and require an EP gradient
    sum once after backward. The independent module does not change the default
    single-rank QwenAir text model.
    """

    def __init__(
        self,
        config: QwenAirTextConfig,
        ep_group: dist.ProcessGroup,
        expert_tp_group: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise ValueError("QwenAir expert parallelism requires initialized torch.distributed")
        if not torch.cuda.is_available() or dist.get_backend(ep_group) != "nccl":
            raise NotImplementedError("MCore QwenAir EP dispatcher currently requires CUDA/NCCL")
        if dist.get_world_size(expert_tp_group) != 1:
            raise NotImplementedError("QwenAir EP prototype requires expert TP size one")
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        if config.num_experts % self.ep_size:
            raise ValueError("QwenAir expert count must divide evenly across EP")
        self.num_local_experts = config.num_experts // self.ep_size
        local_elements = (
            self.num_local_experts * config.moe_intermediate_size * config.hidden_size * 3
        )
        if local_elements > config.max_single_rank_parameters:
            raise ValueError("QwenAir local expert shard exceeds the explicit resource limit")

        # Import the full MCore MoE stack only when the EP prototype is selected.
        # The ordinary single-rank HF reference remains importable without Triton.
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
        from megatron.core.transformer.transformer_config import TransformerConfig

        dispatcher_config = TransformerConfig(
            num_layers=1,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_moe_experts=config.num_experts,
            moe_ffn_hidden_size=config.moe_intermediate_size,
            moe_router_topk=config.num_experts_per_tok,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=self.ep_size,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )
        pg_collection = ProcessGroupCollection(
            ep=ep_group, expt_tp=expert_tp_group, tp_ep=ep_group
        )
        local_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )
        self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            self.num_local_experts,
            local_indices,
            config=dispatcher_config,
            pg_collection=pg_collection,
        )
        self.gate = QwenAirTopKRouter(config)
        self.experts = QwenAirExperts(
            config,
            num_local_experts=self.num_local_experts,
            layer_idx=layer_idx,
            first_global_expert=self.ep_rank * self.num_local_experts,
        )
        # MCore DDP uses ``param.allreduce`` to separate expert shards from
        # parameters replicated across the full data-parallel group.  Without
        # this stamp, equal-shaped shards owned by different EP ranks are
        # incorrectly averaged together.  Expert parameters instead reduce
        # over the expert-data-parallel group supplied to DDP by the trainer.
        for parameter in self.experts.parameters():
            setattr(parameter, "allreduce", self.ep_size == 1)
        self.shared_expert = QwenAirMLP(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Route local token rows to owners and combine routed plus shared output."""
        if not hidden.is_cuda:
            raise ValueError("QwenAir EP dispatcher requires CUDA hidden states")
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits, scores, indices = self.gate(flat)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
        probabilities = torch.zeros_like(logits).scatter(1, indices, scores)
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)

        dispatched, dispatched_scores = self.token_dispatcher.dispatch_preprocess(
            flat, routing_map, probabilities
        )
        dispatched, dispatched_scores = self.token_dispatcher.token_dispatch(
            dispatched, dispatched_scores
        )
        local_hidden, counts, local_scores = self.token_dispatcher.dispatch_postprocess(
            dispatched, dispatched_scores
        )
        local_output = self.experts.forward_dispatched(local_hidden, counts, local_scores)
        local_output = self.token_dispatcher.combine_preprocess(local_output)
        local_output = self.token_dispatcher.token_combine(local_output)
        routed = self.token_dispatcher.combine_postprocess(local_output)
        return (shared + routed).reshape(shape), logits

    def sync_replicated_gradients(self) -> None:
        """Sum router/shared gradients once for standalone EP training tests.

        A trainer using MCore DDP should own this synchronization instead and
        must not call this method a second time.
        """
        for module in (self.gate, self.shared_expert, self.shared_expert_gate):
            for parameter in module.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, group=self.ep_group)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Expose local expert rows as axis-0 shards of the HF logical tensors."""
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"experts.gate_up_proj": 0, "experts.down_proj": 0},
            sharded_offsets,
            tp_group=self.ep_group,
            dp_cp_group=(metadata or {}).get("dp_cp_group"),
        )
