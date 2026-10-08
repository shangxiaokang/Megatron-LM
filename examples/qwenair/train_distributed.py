# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train the QwenAir text model with explicit EP x EDP MCore process groups.

Launch one worker per GPU with ``python -m torch.distributed.run``.  The
default dimensions are a functional gate; pass ``--config-json`` for a real
Qwen4-Exp text configuration.  ``--dry-run`` validates the topology and
reports a parameter-state memory lower bound without initializing CUDA or
allocating model tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.qwenair import (
    QwenAirForCausalLM,
    QwenAirProcessGroups,
    QwenAirTextConfig,
    build_qwenair_process_groups,
    estimate_qwenair_training_memory,
    plan_qwenair_parallel_topology,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer


def tiny_config(expert_model_parallel_size: int, qsa_backend: str) -> QwenAirTextConfig:
    """Return a small model retaining every implemented target block type."""
    return QwenAirTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["qwen_sparse_attention"],
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 0),
        ple_layer_ids=[2],
        ple_embed_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=3,
        qsa_backend=qsa_backend,
        expert_model_parallel_size=expert_model_parallel_size,
        calculate_per_token_loss=True,
    )


def load_config(args: argparse.Namespace) -> QwenAirTextConfig:
    """Load the frozen HF config or the built-in multi-GPU test geometry."""
    if args.config_json is None:
        return tiny_config(args.expert_model_parallel_size, args.qsa_backend or "dense")
    with args.config_json.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    config = QwenAirTextConfig.from_hf_dict(payload)
    updates: dict[str, Any] = {
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "calculate_per_token_loss": True,
    }
    if args.qsa_backend is not None:
        updates["qsa_backend"] = args.qsa_backend
    if args.max_single_rank_ple_elements is not None:
        updates["max_single_rank_ple_elements"] = args.max_single_rank_ple_elements
    if args.max_single_rank_parameters is not None:
        updates["max_single_rank_parameters"] = args.max_single_rank_parameters
    return replace(config, **updates)


def _gib(value: int) -> float:
    return value / 1024**3


def dry_run(config: QwenAirTextConfig, world_size: int) -> dict[str, Any]:
    """Validate a topology and return its allocation-free memory report."""
    topology = plan_qwenair_parallel_topology(config, world_size)
    estimate = estimate_qwenair_training_memory(config, world_size)
    report = {
        "topology": asdict(topology),
        "parameter_counts": {
            "logical": estimate.logical_parameters,
            "replicated_per_rank": estimate.replicated_parameters_per_rank,
            "ple_per_rank": estimate.ple_parameters_per_rank,
            "routed_experts_per_rank": estimate.routed_expert_parameters_per_rank,
        },
        "lower_bound_bytes_per_rank": {
            "bf16_parameters": estimate.bf16_parameter_bytes_per_rank,
            "fp32_gradients": estimate.fp32_gradient_bytes_per_rank,
            "distributed_adam_fp32_master_m_v": estimate.distributed_adam_bytes_per_rank,
            "total": estimate.total_bytes_per_rank,
        },
        "lower_bound_gib_per_rank": {
            "bf16_parameters": _gib(estimate.bf16_parameter_bytes_per_rank),
            "fp32_gradients": _gib(estimate.fp32_gradient_bytes_per_rank),
            "distributed_adam_fp32_master_m_v": _gib(estimate.distributed_adam_bytes_per_rank),
            "total": _gib(estimate.total_bytes_per_rank),
        },
        "excluded": [
            "activations",
            "communication workspaces",
            "allocator fragmentation",
            "checkpoint staging",
        ],
    }
    print(json.dumps(report, indent=2))
    return report


def initialize_distributed() -> torch.device:
    """Initialize the torchrun worker's NCCL process group and CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("QwenAir distributed training requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    return device


def build_model_and_optimizer(
    config: QwenAirTextConfig,
    groups: QwenAirProcessGroups,
    device: torch.device,
    learning_rate: float,
    use_distributed_optimizer: bool,
) -> tuple[DistributedDataParallel, Any]:
    """Construct QwenAir, MCore DDP, and the MCore Adam optimizer."""
    if not config.calculate_per_token_loss:
        raise ValueError("QwenAir MCore DDP training requires calculate_per_token_loss=True")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    module = QwenAirForCausalLM(config, pg_collection=groups.collection).to(
        device=device, dtype=torch.bfloat16
    )
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    model = DistributedDataParallel(
        config=config, ddp_config=ddp_config, module=module, pg_collection=groups.collection
    )
    model.broadcast_params()
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=learning_rate,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    optimizer = get_megatron_optimizer(
        optimizer_config, [model], use_gloo_process_groups=False, pg_collection=groups.collection
    )
    return model, optimizer


def make_batch(
    config: QwenAirTextConfig,
    step: int,
    rank: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Create deterministic rank-distinct token batches for restart checks."""
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + 10_000 * step + rank)
    return torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), generator=generator, device=device
    )


def train_step(
    model: DistributedDataParallel,
    optimizer: Any,
    groups: QwenAirProcessGroups,
    tokens: torch.Tensor,
) -> tuple[float, float]:
    """Run one BF16 forward/backward, gradient synchronization, and Adam step."""
    optimizer.zero_grad()
    model.zero_grad_buffer()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(tokens, labels=tokens, output_router_logits=True)
        if output.loss is None:
            raise RuntimeError("QwenAir training step did not return a loss")
        local_valid_tokens = (tokens[:, 1:] != -100).sum()
        backward_loss, reported_loss = groups.losses_for_mcore_ddp(
            output.loss,
            local_valid_tokens,
            aux_loss=output.aux_loss,
            aux_loss_coefficient=model.module.config.router_aux_loss_coef,
        )
    backward_loss.backward()
    model.finish_grad_sync()
    update_successful, grad_norm, _ = optimizer.step()
    if not update_successful:
        raise RuntimeError("QwenAir optimizer step was skipped")

    grad_norm_value = float(grad_norm) if grad_norm is not None else float("nan")
    return float(reported_loss), grad_norm_value


def _checkpoint_state(
    model: DistributedDataParallel, optimizer: Any, groups: QwenAirProcessGroups, loading: bool
) -> dict[str, Any]:
    optimizer_metadata = {
        "dp_cp_group": groups.collection.dp_cp,
        "distrib_optim_sharding_type": "dp_reshardable",
    }
    model_state = model.module.sharded_state_dict(
        metadata={"dp_cp_group": groups.collection.expt_dp}
    )
    state: dict[str, Any] = {"model": model_state}
    state["optimizer"] = optimizer.sharded_state_dict(
        state, metadata=optimizer_metadata, is_loading=loading
    )
    return state


def save_checkpoint(
    checkpoint_dir: Path,
    model: DistributedDataParallel,
    optimizer: Any,
    groups: QwenAirProcessGroups,
) -> None:
    """Save model shards and distributed Adam state with MCore DCP."""
    if dist.get_rank() == 0 and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    dist.barrier()
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(
        sharded_state_dict=_checkpoint_state(model, optimizer, groups, loading=False),
        checkpoint_dir=str(checkpoint_dir),
    )
    dist.barrier()


def load_checkpoint(
    checkpoint_dir: Path,
    model: DistributedDataParallel,
    optimizer: Any,
    groups: QwenAirProcessGroups,
) -> None:
    """Restore model shards and distributed Adam state from MCore DCP."""
    template = _checkpoint_state(model, optimizer, groups, loading=True)
    loaded = dist_checkpointing.load(
        sharded_state_dict=template, checkpoint_dir=str(checkpoint_dir)
    )
    model.module.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    dist.barrier()


def _snapshot_parameters(model: DistributedDataParallel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.module.named_parameters()
    }


def _assert_parameter_snapshot(
    model: DistributedDataParallel, expected: dict[str, torch.Tensor]
) -> None:
    for name, parameter in model.module.named_parameters():
        torch.testing.assert_close(parameter.detach().cpu(), expected[name], rtol=0, atol=0)


def run_training(args: argparse.Namespace, config: QwenAirTextConfig) -> dict[str, Any]:
    """Execute training, DCP save, and optional exact restart replay."""
    device = initialize_distributed()
    rank = dist.get_rank()
    groups = build_qwenair_process_groups(config)
    model, optimizer = build_model_and_optimizer(
        config, groups, device, args.learning_rate, args.distributed_optimizer
    )
    topology = groups.topology
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": topology.world_size,
                    "expert_model_parallel_size": topology.expert_model_parallel_size,
                    "expert_data_parallel_size": topology.expert_data_parallel_size,
                    "expert_parallel_groups": topology.expert_parallel_groups,
                    "expert_data_parallel_groups": topology.expert_data_parallel_groups,
                },
                indent=2,
            )
        )

    losses: list[float] = []
    grad_norms: list[float] = []
    replay_start = args.checkpoint_step
    if replay_start < 0 or replay_start > args.steps:
        raise ValueError("checkpoint-step must be in [0, steps]")
    if args.verify_restart and not 1 <= replay_start < args.steps:
        raise ValueError("verify-restart requires checkpoint-step in [1, steps)")
    for step in range(args.steps):
        tokens = make_batch(config, step, rank, args.micro_batch_size, args.sequence_length, device)
        loss, grad_norm = train_step(model, optimizer, groups, tokens)
        losses.append(loss)
        grad_norms.append(grad_norm)
        if rank == 0:
            print(f"step={step + 1} loss={loss:.8f} grad_norm={grad_norm:.8f}")
        if replay_start and step + 1 == replay_start:
            save_checkpoint(args.checkpoint_dir, model, optimizer, groups)

    if args.verify_restart:
        expected_losses = losses[replay_start:]
        expected_parameters = _snapshot_parameters(model)
        del optimizer
        del model
        torch.cuda.empty_cache()
        model, optimizer = build_model_and_optimizer(
            config, groups, device, args.learning_rate, args.distributed_optimizer
        )
        load_checkpoint(args.checkpoint_dir, model, optimizer, groups)
        replay_losses = []
        for step in range(replay_start, args.steps):
            tokens = make_batch(
                config, step, rank, args.micro_batch_size, args.sequence_length, device
            )
            loss, _ = train_step(model, optimizer, groups, tokens)
            replay_losses.append(loss)
        torch.testing.assert_close(
            torch.tensor(replay_losses), torch.tensor(expected_losses), rtol=0, atol=0
        )
        _assert_parameter_snapshot(model, expected_parameters)

    result = {
        "world_size": topology.world_size,
        "expert_model_parallel_size": topology.expert_model_parallel_size,
        "expert_data_parallel_size": topology.expert_data_parallel_size,
        "steps": args.steps,
        "losses": losses,
        "grad_norms": grad_norms,
        "distributed_optimizer": args.distributed_optimizer,
        "checkpoint_dir": str(args.checkpoint_dir),
        "restart_verified": args.verify_restart,
    }
    if rank == 0 and args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
    return result


def parse_args() -> argparse.Namespace:
    """Parse functional-gate and target-config training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--expert-model-parallel-size", type=int, default=2)
    parser.add_argument("--qsa-backend", choices=("dense", "te_reference", "te_indexed_sdpa"))
    parser.add_argument("--max-single-rank-ple-elements", type=int)
    parser.add_argument("--max-single-rank-parameters", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--checkpoint-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--distributed-optimizer", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--verify-restart", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("qwenair-checkpoint"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--world-size", type=int)
    return parser.parse_args()


def main() -> None:
    """Run allocation-free planning or distributed training."""
    args = parse_args()
    config = load_config(args)
    if args.dry_run:
        world_size = args.world_size or int(os.environ.get("WORLD_SIZE", "1"))
        dry_run(config, world_size)
        return
    run_training(args, config)


if __name__ == "__main__":
    main()
