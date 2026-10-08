# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Topology, memory planning, and EP x EDP DDP tests for QwenAir."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.qwenair import (
    QwenAirForCausalLM,
    QwenAirTextConfig,
    build_qwenair_process_groups,
    estimate_qwenair_training_memory,
    plan_qwenair_parallel_topology,
)
from megatron.core.models.qwenair.ple import qwenair_ngram_metadata


def _config(*, ep_size: int = 2) -> QwenAirTextConfig:
    return QwenAirTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 0),
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=3,
        qsa_backend="dense",
        expert_model_parallel_size=ep_size,
        calculate_per_token_loss=True,
    )


def test_parallel_topology_and_memory_plan_are_allocation_free():
    """Describe EP2 x EDP2 and categorize every logical parameter."""
    config = _config()
    topology = plan_qwenair_parallel_topology(config, world_size=4)
    assert topology.expert_parallel_groups == ((0, 1), (2, 3))
    assert topology.expert_data_parallel_groups == ((0, 2), (1, 3))
    assert topology.groups_for_rank(3) == ((2, 3), (1, 3))

    estimate = estimate_qwenair_training_memory(config, world_size=4)
    _, _, _, padded_rows = qwenair_ngram_metadata(config)
    expected_ple = padded_rows * (
        config.ple_embed_dim // ((config.ngram_size - 1) * config.heads_per_ngram)
    )
    expected_experts = (
        config.num_hidden_layers
        * config.num_experts
        * 3
        * config.moe_intermediate_size
        * config.hidden_size
    )
    assert estimate.ple_parameters_per_rank == expected_ple // 2
    assert estimate.routed_expert_parameters_per_rank == expected_experts // 2
    assert estimate.logical_parameters > expected_ple + expected_experts
    assert estimate.total_bytes_per_rank > estimate.bf16_parameter_bytes_per_rank


def test_parallel_topology_rejects_illegal_world_before_group_creation():
    """Fail before allocating model tensors or constructing partial process groups."""
    with pytest.raises(ValueError, match="world_size"):
        plan_qwenair_parallel_topology(_config(), world_size=3)
    with pytest.raises(ValueError, match="expert count"):
        _config(ep_size=3)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("tensor_model_parallel_size", "TP=PP=CP=1"),
        ("pipeline_model_parallel_size", "TP=PP=CP=1"),
        ("context_parallel_size", "TP=PP=CP=1"),
        ("expert_tensor_parallel_size", "expert TP size one"),
    ),
)
def test_config_keeps_unimplemented_parallel_axes_explicit(field: str, message: str):
    """Allow EP while retaining precise guards for unsupported axes."""
    values = {field: 2}
    with pytest.raises(NotImplementedError, match=message):
        QwenAirTextConfig(**values)


def _manual_gradient_sync(model: QwenAirForCausalLM, groups) -> None:
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        group = (
            groups.expert_data_parallel_group
            if not getattr(parameter, "allreduce", True)
            else dist.group.WORLD
        )
        dist.all_reduce(parameter.grad, group=group)


def _ddp_oracle_worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    try:
        config = _config()
        groups = build_qwenair_process_groups(config)
        torch.manual_seed(20261008)
        reference = QwenAirForCausalLM(config, pg_collection=groups.collection).to(
            device=device, dtype=torch.bfloat16
        )
        torch.manual_seed(20261008)
        module = QwenAirForCausalLM(config, pg_collection=groups.collection).to(
            device=device, dtype=torch.bfloat16
        )
        module.load_state_dict(reference.state_dict())
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
            ),
            module,
            pg_collection=groups.collection,
        )

        reference_parameters = dict(reference.named_parameters())
        ddp_parameters = dict(module.named_parameters())
        ple_name = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
        expert_name = "model.layers.1.mlp.experts.gate_up_proj"
        dense_name = "lm_head.weight"
        sharded = module.sharded_state_dict(metadata={"dp_cp_group": dist.group.WORLD})
        ep_rank = dist.get_rank(groups.ep_group)
        expert_dp_rank = dist.get_rank(groups.expert_data_parallel_group)
        assert sharded[expert_name].replica_id == (0, 0, expert_dp_rank)
        assert sharded[dense_name].replica_id == (0, ep_rank, expert_dp_rank)
        assert not getattr(reference_parameters[ple_name], "allreduce", True)
        assert not getattr(reference_parameters[expert_name], "allreduce", True)
        assert getattr(reference_parameters[dense_name], "allreduce", True)

        generator = torch.Generator(device=device).manual_seed(1000 + rank)
        tokens = torch.randint(
            4, config.vocab_size, (1, 9 + rank), generator=generator, device=device
        )
        tokens[:, 3] = config.eos_token_id

        reference.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reference_output = reference(tokens, labels=tokens, output_router_logits=True)
            reference_loss, _ = groups.losses_for_mcore_ddp(
                reference_output.loss,
                tokens.shape[1] - 1,
                aux_loss=reference_output.aux_loss,
                aux_loss_coefficient=config.router_aux_loss_coef,
            )
        reference_loss.backward()
        _manual_gradient_sync(reference, groups)

        ddp.zero_grad_buffer()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ddp_output = ddp(tokens, labels=tokens, output_router_logits=True)
            ddp_loss, _ = groups.losses_for_mcore_ddp(
                ddp_output.loss,
                tokens.shape[1] - 1,
                aux_loss=ddp_output.aux_loss,
                aux_loss_coefficient=config.router_aux_loss_coef,
            )
        ddp_loss.backward()
        ddp.finish_grad_sync()

        for name in (dense_name, expert_name, ple_name):
            expected = reference_parameters[name].grad.float()
            actual = ddp_parameters[name].main_grad
            # A routed expert may receive no tokens in a small random batch.  Its
            # zero gradient must still agree with the explicit EDP reduction.
            if name != expert_name:
                assert expected.abs().sum() > 0, name
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=3e-4, msg=name)

        dense_bucket_parameters = {
            parameter
            for buffer in ddp.buffers
            for bucket in buffer.buckets
            for parameter in bucket.params
        }
        expert_bucket_parameters = {
            parameter
            for buffer in ddp.expert_parallel_buffers
            for bucket in buffer.buckets
            for parameter in bucket.params
        }
        assert ddp_parameters[dense_name] in dense_bucket_parameters
        assert ddp_parameters[expert_name] in expert_bucket_parameters
        assert ddp_parameters[ple_name] in expert_bucket_parameters
    finally:
        dist.destroy_process_group()


def test_four_rank_full_model_gradients_match_ep2_edp2_oracle():
    """Match dense, routed-expert, and PLE gradients against explicit reductions."""
    if torch.cuda.device_count() < 4 or not dist.is_nccl_available():
        pytest.skip("Four CUDA devices and NCCL are required for the EP2 x EDP2 test")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    mp.spawn(_ddp_oracle_worker, args=(port,), nprocs=4, join=True)
