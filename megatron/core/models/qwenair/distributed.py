# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit EP x EDP topology and memory planning for QwenAir training."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection

from .config import QwenAirTextConfig
from .ple import qwenair_ngram_metadata


@dataclass(frozen=True)
class QwenAirParallelTopology:
    """Rank layout for QwenAir expert parallelism and its replicas."""

    world_size: int
    expert_model_parallel_size: int
    expert_data_parallel_size: int
    expert_parallel_groups: tuple[tuple[int, ...], ...]
    expert_data_parallel_groups: tuple[tuple[int, ...], ...]

    def groups_for_rank(self, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the EP and expert-DP rank tuples containing ``rank``."""
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        ep_group = self.expert_parallel_groups[rank // self.expert_model_parallel_size]
        expert_dp_group = self.expert_data_parallel_groups[rank % self.expert_model_parallel_size]
        return ep_group, expert_dp_group


@dataclass(frozen=True)
class QwenAirProcessGroups:
    """Materialized QwenAir groups passed explicitly to model, DDP, and optimizer."""

    topology: QwenAirParallelTopology
    collection: ProcessGroupCollection

    @property
    def ep_group(self) -> dist.ProcessGroup:
        """Process group that owns one complete set of routed experts and PLE rows."""
        return self.collection.ep

    @property
    def expert_data_parallel_group(self) -> dist.ProcessGroup:
        """Process group containing replicas of the same local expert/PLE shard."""
        return self.collection.expt_dp

    @property
    def expert_tp_group(self) -> dist.ProcessGroup:
        """Singleton expert tensor-parallel group for the current implementation."""
        return self.collection.expt_tp

    def losses_for_mcore_ddp(
        self,
        loss: Tensor,
        local_valid_tokens: Tensor | int,
        *,
        aux_loss: Tensor | None = None,
        aux_loss_coefficient: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Return correctly scaled backward and globally reported losses.

        The model's CE numerator is local while its denominator is the valid
        token count across EP. Replicas may contain different token counts, so
        a fixed ``1 / EDP`` multiplier would average replica means instead of
        producing a global token mean. This method weights CE by
        ``EP tokens / WORLD tokens``. The router auxiliary objective is one
        objective per EP replica and is averaged across EDP; its differentiable
        EP reduction already compensates for the later dense-gradient sum.

        ``config.calculate_per_token_loss`` must be true so MCore DDP sums the
        resulting dense WORLD gradients and expert/PLE EDP gradients without
        applying a second scale factor.
        """
        if isinstance(local_valid_tokens, Tensor):
            local_count = local_valid_tokens.detach().to(device=loss.device, dtype=torch.float32)
        else:
            local_count = torch.tensor(local_valid_tokens, device=loss.device, dtype=torch.float32)
        ep_count = local_count.clone()
        dist.all_reduce(ep_count, group=self.ep_group)
        world_count = ep_count.clone()
        dist.all_reduce(world_count, group=self.expert_data_parallel_group)
        if world_count.item() <= 0:
            raise ValueError("QwenAir distributed loss requires at least one valid token")

        ce_loss = loss
        if aux_loss is not None:
            ce_loss = ce_loss - aux_loss_coefficient * aux_loss
        token_weight = ep_count / world_count
        backward_loss = ce_loss * token_weight
        reporting_loss = ce_loss.detach() * token_weight
        if aux_loss is not None and aux_loss_coefficient:
            backward_loss = backward_loss + (
                aux_loss_coefficient * aux_loss / self.topology.expert_data_parallel_size
            )
            reporting_loss = reporting_loss + (
                aux_loss_coefficient
                * aux_loss.detach()
                / (
                    self.topology.expert_data_parallel_size
                    * self.topology.expert_model_parallel_size
                )
            )
        dist.all_reduce(reporting_loss, group=dist.group.WORLD)
        return backward_loss, reporting_loss


@dataclass(frozen=True)
class QwenAirMemoryEstimate:
    """Per-rank parameter and optimizer-state lower bound in bytes."""

    logical_parameters: int
    replicated_parameters_per_rank: int
    ple_parameters_per_rank: int
    routed_expert_parameters_per_rank: int
    bf16_parameter_bytes_per_rank: int
    fp32_gradient_bytes_per_rank: int
    distributed_adam_bytes_per_rank: int
    total_bytes_per_rank: int


def plan_qwenair_parallel_topology(
    config: QwenAirTextConfig, world_size: int
) -> QwenAirParallelTopology:
    """Validate and describe the supported contiguous EP x EDP rank layout.

    EP groups are contiguous so each replica can route locally.  Expert-DP
    groups join the same EP coordinate across replicas.  Dense parameters use
    the full world as their data-parallel group.
    """
    if world_size < 1:
        raise ValueError("QwenAir world_size must be positive")
    ep_size = config.expert_model_parallel_size
    if world_size % ep_size:
        raise ValueError("QwenAir world_size must divide evenly by expert parallel size")
    if config.num_experts % ep_size:
        raise ValueError("QwenAir expert count must divide evenly across EP")
    expert_dp_size = world_size // ep_size
    expert_parallel_groups = tuple(
        tuple(range(replica * ep_size, (replica + 1) * ep_size))
        for replica in range(expert_dp_size)
    )
    expert_data_parallel_groups = tuple(
        tuple(replica * ep_size + ep_rank for replica in range(expert_dp_size))
        for ep_rank in range(ep_size)
    )
    ple_elements_per_rank = 0
    head_width = config.ple_embed_dim // ((config.ngram_size - 1) * config.heads_per_ngram)
    for ple_index in range(len(config.ple_layer_ids)):
        _, _, _, padded_rows = qwenair_ngram_metadata(config, ple_index)
        if padded_rows % ep_size:
            raise ValueError("QwenAir padded PLE rows must divide evenly across EP")
        ple_elements_per_rank += padded_rows // ep_size * head_width
    if ple_elements_per_rank > config.max_single_rank_ple_elements:
        raise ValueError(
            "QwenAir local PLE shard exceeds max_single_rank_ple_elements; "
            "increase EP or explicitly raise the resource limit"
        )
    routed_expert_parameters_per_rank = (
        config.num_hidden_layers
        * (config.num_experts // ep_size)
        * 3
        * config.moe_intermediate_size
        * config.hidden_size
    )
    if routed_expert_parameters_per_rank > config.max_single_rank_parameters:
        raise ValueError(
            "QwenAir local expert shard exceeds max_single_rank_parameters; "
            "increase EP or explicitly raise the resource limit"
        )
    return QwenAirParallelTopology(
        world_size=world_size,
        expert_model_parallel_size=ep_size,
        expert_data_parallel_size=expert_dp_size,
        expert_parallel_groups=expert_parallel_groups,
        expert_data_parallel_groups=expert_data_parallel_groups,
    )


def _new_groups_for_rank(groups: tuple[tuple[int, ...], ...], rank: int) -> dist.ProcessGroup:
    """Create every group in deterministic order and return this rank's group."""
    local_group = None
    for ranks in groups:
        process_group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            local_group = process_group
    if local_group is None:
        raise RuntimeError(f"rank {rank} was not assigned to a process group")
    return local_group


def build_qwenair_process_groups(config: QwenAirTextConfig) -> QwenAirProcessGroups:
    """Materialize explicit groups for QwenAir EP x EDP MCore training."""
    if not dist.is_initialized():
        raise ValueError("QwenAir process groups require initialized torch.distributed")
    rank = dist.get_rank()
    topology = plan_qwenair_parallel_topology(config, dist.get_world_size())

    singleton_groups = tuple((group_rank,) for group_rank in range(topology.world_size))
    singleton = _new_groups_for_rank(singleton_groups, rank)
    ep_group = _new_groups_for_rank(topology.expert_parallel_groups, rank)
    expert_dp_group = _new_groups_for_rank(topology.expert_data_parallel_groups, rank)
    world = dist.group.WORLD
    collection = ProcessGroupCollection(
        tp=singleton,
        pp=singleton,
        mp=singleton,
        cp=singleton,
        tp_cp=singleton,
        ep=ep_group,
        expt_tp=singleton,
        tp_ep=ep_group,
        tp_ep_pp=ep_group,
        dp=world,
        dp_cp=world,
        tp_dp_cp=world,
        expt_dp=expert_dp_group,
        intra_dp_cp=world,
        intra_expt_dp=expert_dp_group,
        intra_dist_opt=world,
        inter_dist_opt=None,
    )
    return QwenAirProcessGroups(topology=topology, collection=collection)


def estimate_qwenair_training_memory(
    config: QwenAirTextConfig, world_size: int
) -> QwenAirMemoryEstimate:
    """Estimate per-rank BF16, FP32-gradient, and Adam storage without allocation.

    The estimate excludes activations, temporary communication buffers,
    allocator fragmentation, and checkpoint staging.  Adam counts FP32 master
    weights plus two FP32 moments, sharded over WORLD for replicated parameters
    and over EDP for each local expert/PLE shard.
    """
    topology = plan_qwenair_parallel_topology(config, world_size)
    ple_parameters = 0
    head_width = config.ple_embed_dim // ((config.ngram_size - 1) * config.heads_per_ngram)
    for ple_index in range(len(config.ple_layer_ids)):
        _, _, _, padded_rows = qwenair_ngram_metadata(config, ple_index)
        ple_parameters += padded_rows * head_width
    routed_expert_parameters = (
        config.num_hidden_layers
        * config.num_experts
        * 3
        * config.moe_intermediate_size
        * config.hidden_size
    )

    # Meta construction gives an exact count for the replicated architecture
    # without materializing the target model's tensors.
    logical_config = replace(
        config,
        expert_model_parallel_size=1,
        max_single_rank_ple_elements=max(config.max_single_rank_ple_elements, ple_parameters),
        max_single_rank_parameters=max(config.max_single_rank_parameters, routed_expert_parameters),
    )
    from .model import QwenAirForCausalLM

    with torch.device("meta"):
        logical_model = QwenAirForCausalLM(logical_config)
    logical_parameters = sum(parameter.numel() for parameter in logical_model.parameters())
    del logical_model
    replicated_parameters = logical_parameters - ple_parameters - routed_expert_parameters
    if replicated_parameters < 0:
        raise RuntimeError("QwenAir parameter categorization produced a negative dense count")

    ple_per_rank = ple_parameters // topology.expert_model_parallel_size
    routed_per_rank = routed_expert_parameters // topology.expert_model_parallel_size
    local_parameters = replicated_parameters + ple_per_rank + routed_per_rank
    bf16_parameter_bytes = 2 * local_parameters
    fp32_gradient_bytes = 4 * local_parameters
    dense_adam_parameters = (replicated_parameters + topology.world_size - 1) // topology.world_size
    sharded_adam_parameters = (
        ple_per_rank + routed_per_rank + topology.expert_data_parallel_size - 1
    ) // topology.expert_data_parallel_size
    distributed_adam_bytes = 12 * (dense_adam_parameters + sharded_adam_parameters)
    return QwenAirMemoryEstimate(
        logical_parameters=logical_parameters,
        replicated_parameters_per_rank=replicated_parameters,
        ple_parameters_per_rank=ple_per_rank,
        routed_expert_parameters_per_rank=routed_per_rank,
        bf16_parameter_bytes_per_rank=bf16_parameter_bytes,
        fp32_gradient_bytes_per_rank=fp32_gradient_bytes,
        distributed_adam_bytes_per_rank=distributed_adam_bytes,
        total_bytes_per_rank=(bf16_parameter_bytes + fp32_gradient_bytes + distributed_adam_bytes),
    )
