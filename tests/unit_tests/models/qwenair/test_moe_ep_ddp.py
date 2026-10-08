# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MCore DDP ownership checks for QwenAir expert-parallel parameters."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.qwenair import QwenAirTextConfig
from megatron.core.models.qwenair.moe_ep import QwenAirExpertParallelBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


def _config() -> QwenAirTextConfig:
    return QwenAirTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        hc_count=4,
        hc_lowrank=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 0),
        ple_layer_ids=[],
    )


def _groups(rank: int):
    singleton_groups = [dist.new_group([group_rank]) for group_rank in range(4)]
    ep_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
    expert_dp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    return (
        singleton_groups[rank],
        ep_groups[rank // 2],
        expert_dp_groups[rank % 2],
    )


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        singleton_group, ep_group, expert_dp_group = _groups(rank)
        model = QwenAirExpertParallelBlock(_config(), ep_group, singleton_group).to(device)

        expert_parameters = set(model.experts.parameters())
        assert expert_parameters
        assert all(not getattr(parameter, "allreduce", True) for parameter in expert_parameters)
        assert all(
            getattr(parameter, "allreduce", True)
            for parameter in model.parameters()
            if parameter not in expert_parameters
        )

        pg_collection = ProcessGroupCollection(
            dp=dist.group.WORLD,
            dp_cp=dist.group.WORLD,
            expt_dp=expert_dp_group,
            tp=singleton_group,
            pp=singleton_group,
            ep=ep_group,
        )
        ddp = DistributedDataParallel(
            TransformerConfig(
                num_layers=1,
                hidden_size=16,
                num_attention_heads=2,
                num_moe_experts=4,
                moe_ffn_hidden_size=8,
                expert_model_parallel_size=2,
            ),
            DistributedDataParallelConfig(overlap_grad_reduce=False),
            model,
            pg_collection=pg_collection,
        )
        assert len(ddp.buffers) == 1
        assert len(ddp.expert_parallel_buffers) == 1
        assert {
            parameter
            for buffer in ddp.expert_parallel_buffers
            for bucket in buffer.buckets
            for parameter in bucket.params
        } == expert_parameters

        # Give each EP/EDP coordinate distinct tokens.  Dense parameters must
        # reduce over all four ranks; a local expert shard must only reduce with
        # the rank holding the same shard in the other EP replica.
        ddp.zero_grad_buffer()
        torch.manual_seed(100 + rank)
        hidden = torch.randn(3 + rank, 16, device=device)
        output, logits = ddp(hidden)
        (output.float().square().mean() + logits.float().square().mean()).backward()
        ddp.finish_grad_sync()

        dense_grad = model.gate.weight.main_grad
        dense_reference = dense_grad.clone()
        dist.broadcast(dense_reference, src=0)
        torch.testing.assert_close(dense_grad, dense_reference)

        expert_grad = model.experts.gate_up_proj.main_grad
        expert_reference = expert_grad.clone()
        source = 0 if rank % 2 == 0 else 1
        dist.broadcast(expert_reference, src=source, group=expert_dp_group)
        torch.testing.assert_close(expert_grad, expert_reference)
    finally:
        dist.destroy_process_group()


def test_four_rank_qwenair_ep_ddp_uses_expert_replica_groups():
    """Keep routed shards out of the dense DP all-reduce."""
    if torch.cuda.device_count() < 4 or not dist.is_nccl_available():
        pytest.skip("Four CUDA devices and NCCL are required for the EP x EDP DDP test")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)
