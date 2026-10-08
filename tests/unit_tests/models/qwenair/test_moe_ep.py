# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir local expert parity and two-rank MCore all-to-all EP prototype."""

from __future__ import annotations

import importlib.util
import socket
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.models.qwenair import QwenAirTextConfig
from megatron.core.models.qwenair.layers import (
    QwenAirExperts,
    QwenAirSparseMoeBlock,
    qwenair_global_router_loss,
)
from megatron.core.models.qwenair.moe_ep import QwenAirExpertParallelBlock, qwenair_ep_router_loss


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


def test_local_packed_experts_match_expert_sorted_dispatch():
    """MCore's contiguous expert order preserves the single-rank HF math."""
    torch.manual_seed(37)
    experts = QwenAirExperts(_config(), num_local_experts=2)
    reference = QwenAirExperts(_config(), num_local_experts=2)
    reference.load_state_dict(experts.state_dict(), strict=True)
    input_rows = torch.randn(5, 16)
    scores = torch.tensor([0.2, 0.5, 0.7, 0.3, 0.9])
    counts = torch.tensor([3, 2])
    indices = torch.tensor([[0], [0], [0], [1], [1]])

    actual = experts.forward_dispatched(input_rows, counts, scores)
    expected = reference(input_rows, indices, scores.unsqueeze(-1))
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(experts.gate_up_proj.grad, reference.gate_up_proj.grad)
    torch.testing.assert_close(experts.down_proj.grad, reference.down_proj.grad)


def _aux_worker(rank: int, port: int) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        all_logits = torch.tensor(
            [[0.1, 1.0, -0.7, 0.3], [1.2, -0.2, 0.4, 0.6],
             [0.2, 0.3, 0.4, 1.3], [-0.3, 0.8, 0.7, 0.1],
             [0.5, 0.1, 1.1, -0.2]],
            requires_grad=True,
        )
        start, stop = (0, 2) if rank == 0 else (2, 5)
        local_logits = all_logits.detach()[start:stop].clone().requires_grad_()
        expected = qwenair_global_router_loss((all_logits,), 4, 2)
        actual = qwenair_ep_router_loss((local_logits,), 4, 2, dist.group.WORLD)
        torch.testing.assert_close(actual, expected)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(local_logits.grad, all_logits.grad[start:stop])
    finally:
        dist.destroy_process_group()


def test_two_rank_global_router_loss_preserves_dense_gradient():
    """The EP all-reduce returns the global HF loss and one global gradient."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    mp.spawn(_aux_worker, args=(port,), nprocs=2, join=True)


def _copy_full_weights(reference, parallel, rank, local_experts):
    with torch.no_grad():
        parallel.gate.load_state_dict(reference.gate.state_dict(), strict=True)
        parallel.shared_expert.load_state_dict(reference.shared_expert.state_dict(), strict=True)
        parallel.shared_expert_gate.load_state_dict(
            reference.shared_expert_gate.state_dict(), strict=True
        )
        start = rank * local_experts
        stop = start + local_experts
        parallel.experts.gate_up_proj.copy_(reference.experts.gate_up_proj[start:stop])
        parallel.experts.down_proj.copy_(reference.experts.down_proj[start:stop])


def _check_empty_source_rank(reference, parallel, rank, config, device):
    """A rank with no local tokens still receives and trains its owned experts."""
    reference.zero_grad(set_to_none=True)
    parallel.zero_grad(set_to_none=True)
    full_input = torch.sin(
        torch.arange(9 * config.hidden_size, device=device).float().reshape(9, -1) * 0.37
    ).requires_grad_()
    local_input = full_input.detach()[:0 if rank == 0 else 9].clone().requires_grad_()
    expected, expected_logits = reference(full_input)
    actual, actual_logits = parallel(local_input)
    torch.testing.assert_close(actual, expected[:0] if rank == 0 else expected)
    torch.testing.assert_close(actual_logits, expected_logits[:0] if rank == 0 else expected_logits)
    expected_aux = qwenair_global_router_loss(
        (expected_logits,), config.num_experts, config.num_experts_per_tok
    )
    actual_aux = qwenair_ep_router_loss(
        (actual_logits,), config.num_experts, config.num_experts_per_tok, dist.group.WORLD
    )
    torch.testing.assert_close(actual_aux, expected_aux)
    (expected.square().sum() + expected_aux).backward()
    (actual.square().sum() + actual_aux).backward()
    parallel.sync_replicated_gradients()
    if rank == 1:
        torch.testing.assert_close(local_input.grad, full_input.grad, rtol=1e-4, atol=1e-5)
    for name in ("gate_up_proj", "down_proj"):
        actual_grad = getattr(parallel.experts, name).grad
        expected_grad = getattr(reference.experts, name).grad
        assert actual_grad is not None and expected_grad is not None
        start = rank * parallel.num_local_experts
        stop = start + parallel.num_local_experts
        torch.testing.assert_close(
            actual_grad, expected_grad[start:stop], rtol=1e-4, atol=1e-5, msg=name
        )
    for module_name in ("gate", "shared_expert", "shared_expert_gate"):
        actual_module = getattr(parallel, module_name)
        reference_module = getattr(reference, module_name)
        for (name, actual_parameter), (_, expected_parameter) in zip(
            actual_module.named_parameters(), reference_module.named_parameters()
        ):
            torch.testing.assert_close(
                actual_parameter.grad, expected_parameter.grad,
                rtol=1e-4, atol=1e-5, msg=f"{module_name}.{name}",
            )


def _worker(rank: int, port: int, checkpoint_dir: str | None) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        timeout=timedelta(seconds=120),
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        singleton_groups = [dist.new_group(ranks=[group_rank]) for group_rank in range(2)]
        config = _config()
        torch.manual_seed(39)
        reference = QwenAirSparseMoeBlock(config).to(device)
        with torch.no_grad():
            reference.gate.weight.zero_()
            for expert in range(config.num_experts):
                reference.gate.weight[expert, expert] = 5
                reference.gate.weight[expert, (expert + 1) % config.num_experts] = 0.25
        parallel = QwenAirExpertParallelBlock(
            config, dist.group.WORLD, singleton_groups[rank]
        ).to(device)
        assert parallel.num_local_experts == config.num_experts // 2
        _copy_full_weights(reference, parallel, rank, parallel.num_local_experts)
        assert set(parallel.state_dict()) == set(reference.state_dict())
        assert parallel.experts.gate_up_proj.shape[0] == config.num_experts // 2

        expert_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device)
        full_input = torch.zeros(8, config.hidden_size, device=device)
        full_input.scatter_(1, expert_ids.unsqueeze(1), 1.0)
        full_input += torch.arange(config.hidden_size, device=device).float() * 0.001
        full_input.requires_grad_()
        start, stop = (0, 3) if rank == 0 else (3, 8)
        local_input = full_input.detach()[start:stop].clone().requires_grad_()

        reference_output, reference_logits = reference(full_input)
        parallel_output, parallel_logits = parallel(local_input)
        torch.testing.assert_close(
            parallel_output, reference_output[start:stop], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(parallel_logits, reference_logits[start:stop])
        reference_aux = qwenair_global_router_loss(
            (reference_logits,), config.num_experts, config.num_experts_per_tok
        )
        parallel_aux = qwenair_ep_router_loss(
            (parallel_logits,), config.num_experts, config.num_experts_per_tok, dist.group.WORLD
        )
        torch.testing.assert_close(parallel_aux, reference_aux, rtol=1e-6, atol=1e-7)

        (reference_output.square().sum() + reference_aux).backward()
        (parallel_output.square().sum() + parallel_aux).backward()
        parallel.sync_replicated_gradients()
        torch.testing.assert_close(
            local_input.grad, full_input.grad[start:stop], rtol=1e-4, atol=1e-5
        )
        for name in ("gate_up_proj", "down_proj"):
            actual = getattr(parallel.experts, name).grad
            expected = getattr(reference.experts, name).grad
            torch.testing.assert_close(
                actual, expected[rank * 2 : (rank + 1) * 2], rtol=1e-4, atol=1e-5
            )
        for module_name in ("gate", "shared_expert", "shared_expert_gate"):
            actual_module = getattr(parallel, module_name)
            reference_module = getattr(reference, module_name)
            for (name, actual), (_, expected) in zip(
                actual_module.named_parameters(), reference_module.named_parameters()
            ):
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=1e-4, atol=1e-5, msg=name
                )

        descriptor = parallel.sharded_state_dict(prefix="moe.", metadata={"dp_cp_group": None})
        for name in ("gate_up_proj", "down_proj"):
            shard = descriptor[f"moe.experts.{name}"]
            assert shard.global_shape[0] == config.num_experts
            assert shard.global_offset[0] == rank * parallel.num_local_experts
        assert descriptor["moe.gate.weight"].global_offset == (0, 0)
        if checkpoint_dir is not None:
            from megatron.core.dist_checkpointing import load as load_sharded
            from megatron.core.dist_checkpointing import save as save_sharded

            save_sharded(descriptor, checkpoint_dir)
            dist.barrier()
            reloaded = QwenAirExpertParallelBlock(
                config, dist.group.WORLD, singleton_groups[rank]
            ).to(device)
            loaded = load_sharded(
                reloaded.sharded_state_dict(prefix="moe.", metadata={"dp_cp_group": None}),
                checkpoint_dir,
            )
            reloaded.load_state_dict(
                {name.removeprefix("moe."): value for name, value in loaded.items()}, strict=True
            )
            for name, tensor in parallel.state_dict().items():
                torch.testing.assert_close(tensor, reloaded.state_dict()[name])
        _check_empty_source_rank(reference, parallel, rank, config, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        dist.destroy_process_group()


def test_two_rank_qwenair_expert_parallel_matches_full_reference():
    """Check variable token counts, gradients, global aux, and MCore checkpoint."""
    import pytest

    if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip("Two CUDA devices and NCCL are required for MCore EP dispatcher")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    if importlib.util.find_spec("psutil") is None:
        mp.spawn(_worker, args=(port, None), nprocs=2, join=True)
    else:
        workspace = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory(prefix="qwenair-moe-ep-", dir=workspace) as root:
            checkpoint_dir = Path(root) / "sharded"
            checkpoint_dir.mkdir()
            mp.spawn(_worker, args=(port, str(checkpoint_dir)), nprocs=2, join=True)
