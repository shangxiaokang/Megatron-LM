# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-GPU QwenAir text training with PLE and expert parallelism together.

This tiny test checks the logical HF-compatible model against an unsharded
reference. It also checks the combined MCore PLE/expert checkpoint schema and
exact continuation of an AdamW step after a distributed model reload.
"""

from __future__ import annotations

import copy
import socket
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirTextConfig
from megatron.core.models.qwenair.layers import qwenair_global_router_loss


def _config() -> QwenAirTextConfig:
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
        router_aux_loss_coef=0.1,
        qsa_backend="dense",
    )


def _tokens(rank: int, step: int, device: torch.device) -> torch.Tensor:
    length = (9, 13)[rank] + step
    tokens = ((torch.arange(length, device=device) * 7 + rank * 11 + step * 5) % 60 + 4)
    tokens[3] = 3  # Exercise the PLE EOS reset on both unequal local sequences.
    return tokens.unsqueeze(0)


def _logical_slice(model: QwenAirForCausalLM, name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
        layer = int(name.split(".")[2])
        table = model.model.layers[layer].ple.ple_embedding
        return tensor[table.shard_start : table.shard_end]
    if ".mlp.experts." in name:
        local_experts = model.model.layers[0].mlp.num_local_experts
        start = model.model.layers[0].mlp.ep_rank * local_experts
        return tensor[start : start + local_experts]
    return tensor


def _copy_logical_weights(reference: QwenAirForCausalLM, parallel: QwenAirForCausalLM) -> None:
    source = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in parallel.named_parameters():
            parameter.copy_(_logical_slice(parallel, name, source[name]))


def _global_reference_aux(
    router_logits: tuple[torch.Tensor, ...], config: QwenAirTextConfig, lengths: tuple[int, int]
) -> torch.Tensor:
    rank = dist.get_rank()
    gathered_layers = []
    for logits in router_logits:
        padded = F.pad(logits.detach(), (0, 0, 0, max(lengths) - logits.shape[0]))
        received = [torch.empty_like(padded) for _ in lengths]
        dist.all_gather(received, padded)
        global_logits = torch.cat(
            [logits if peer == rank else received[peer][:length].detach()
             for peer, length in enumerate(lengths)], dim=0
        )
        gathered_layers.append(global_logits)
    return qwenair_global_router_loss(
        tuple(gathered_layers), config.num_experts, config.num_experts_per_tok
    )


def _local_ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted = logits[:, :-1].float().contiguous()
    return F.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]), labels[:, 1:].reshape(-1), reduction="sum"
    )


def _reference_step(
    reference: QwenAirForCausalLM,
    parallel: QwenAirForCausalLM,
    reference_optimizer: torch.optim.Optimizer,
    parallel_optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    config: QwenAirTextConfig,
    lengths: tuple[int, int],
    *,
    bf16: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reference_optimizer.zero_grad(set_to_none=True)
    parallel_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        expected = reference(tokens, output_router_logits=True)
        actual = parallel(tokens, labels=tokens, output_router_logits=True)
    assert expected.router_logits is not None and actual.router_logits is not None
    assert actual.loss is not None and actual.aux_loss is not None
    tolerance = {"rtol": 2e-2, "atol": 1e-3} if bf16 else {"rtol": 2e-4, "atol": 2e-5}
    torch.testing.assert_close(actual.logits, expected.logits, **tolerance)
    for actual_router, expected_router in zip(actual.router_logits, expected.router_logits):
        torch.testing.assert_close(actual_router, expected_router, **tolerance)

    total_labels = torch.tensor(sum(length - 1 for length in lengths), device=tokens.device)
    reference_aux = _global_reference_aux(expected.router_logits, config, lengths)
    torch.testing.assert_close(actual.aux_loss, reference_aux, rtol=2e-4, atol=2e-5)
    reference_loss = _local_ce_sum(expected.logits, tokens) / total_labels
    reference_loss = reference_loss + config.router_aux_loss_coef * reference_aux
    actual_ce = _local_ce_sum(actual.logits, tokens) / total_labels
    torch.testing.assert_close(
        actual.loss, actual_ce + config.router_aux_loss_coef * actual.aux_loss,
        rtol=2e-4, atol=2e-5,
    )
    reference_loss.backward()
    actual.loss.backward()
    for parameter in reference.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
    parallel.sync_ep_replicated_gradients()

    reference_params = dict(reference.named_parameters())
    gradients = {}
    for name, parameter in parallel.named_parameters():
        expected_grad = reference_params[name].grad
        if expected_grad is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, _logical_slice(parallel, name, expected_grad),
            rtol=0.08 if bf16 else 3e-3, atol=2e-4 if bf16 else 2e-6, msg=name,
        )
        gradients[name] = parameter.grad.detach().clone()
    assert gradients["model.layers.0.ple.ple_embedding.ngram_embedding.weight"].abs().sum() > 0
    assert gradients["model.layers.0.linear_attn.in_proj_qkv.weight"].abs().sum() > 0
    assert gradients["model.layers.1.self_attn.q_proj.weight"].abs().sum() > 0
    assert gradients["model.layers.1.mlp.experts.gate_up_proj"].abs().sum() > 0
    assert gradients["lm_head.weight"].abs().sum() > 0
    assert "model.layers.1.self_attn.indexer.index_qk_proj.weight" not in gradients

    reference_optimizer.step()
    parallel_optimizer.step()
    for name, parameter in parallel.named_parameters():
        torch.testing.assert_close(
            parameter, _logical_slice(parallel, name, reference_params[name]),
            rtol=0.08 if bf16 else 3e-3, atol=2e-4 if bf16 else 2e-6, msg=name,
        )
    return actual.loss.detach().clone(), gradients


def _check_combined_shards(model: QwenAirForCausalLM, rank: int, config: QwenAirTextConfig):
    sharded = model.sharded_state_dict(prefix="qwenair.", metadata={"dp_cp_group": None})
    table_name = "qwenair.model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    table = model.model.layers[0].ple.ple_embedding
    assert sharded[table_name].global_shape[0] == table.padded_rows
    assert sharded[table_name].global_offset[0] == table.shard_start
    for layer in range(config.num_hidden_layers):
        for projection in ("gate_up_proj", "down_proj"):
            key = f"qwenair.model.layers.{layer}.mlp.experts.{projection}"
            assert sharded[key].global_shape[0] == config.num_experts
            assert sharded[key].global_offset[0] == rank * config.num_experts // 2
    assert sharded["qwenair.model.embed_tokens.weight"].global_offset == (0, 0)
    return sharded


def _worker(rank: int, port: int, checkpoint_dir: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        timeout=timedelta(seconds=180),
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        from megatron.core.dist_checkpointing import load as load_sharded
        from megatron.core.dist_checkpointing import save as save_sharded

        singleton_groups = [dist.new_group(ranks=[group_rank]) for group_rank in range(2)]
        config = _config()
        torch.manual_seed(20261008)
        reference = QwenAirForCausalLM(config).to(device).train()
        with torch.no_grad():
            for layer in reference.model.layers:
                layer.mlp.gate.weight.normal_(std=0.2)
        parallel = QwenAirForCausalLM(
            config, ple_process_group=dist.group.WORLD, ep_group=dist.group.WORLD,
            expert_tp_group=singleton_groups[rank],
        ).to(device).train()
        _copy_logical_weights(reference, parallel)
        assert set(reference.state_dict()) == set(parallel.state_dict())
        _check_combined_shards(parallel, rank, config)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        parallel_optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-3)

        _reference_step(
            reference, parallel, reference_optimizer, parallel_optimizer,
            _tokens(rank, 0, device), config, (9, 13), bf16=False,
        )
        save_sharded(
            parallel.sharded_state_dict(prefix="qwenair.", metadata={"dp_cp_group": None}),
            checkpoint_dir,
        )
        optimizer_path = Path(checkpoint_dir).parent / f"optimizer-rank-{rank}.pt"
        torch.save(parallel_optimizer.state_dict(), optimizer_path)
        reference_checkpoint = {
            "model": copy.deepcopy(reference.state_dict()),
            "optimizer": copy.deepcopy(reference_optimizer.state_dict()),
        }
        dist.barrier()

        tokens = _tokens(rank, 1, device)
        expected_loss, expected_gradients = _reference_step(
            reference, parallel, reference_optimizer, parallel_optimizer,
            tokens, config, (10, 14), bf16=True,
        )
        restored = QwenAirForCausalLM(
            config, ple_process_group=dist.group.WORLD, ep_group=dist.group.WORLD,
            expert_tp_group=singleton_groups[rank],
        ).to(device).train()
        restored_state = load_sharded(
            restored.sharded_state_dict(prefix="qwenair.", metadata={"dp_cp_group": None}),
            checkpoint_dir,
        )
        restored.load_state_dict(
            {name.removeprefix("qwenair."): value for name, value in restored_state.items()},
            strict=True,
        )
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        reference.load_state_dict(reference_checkpoint["model"], strict=True)
        reference_optimizer.load_state_dict(reference_checkpoint["optimizer"])
        replay_loss, replay_gradients = _reference_step(
            reference, restored, reference_optimizer, restored_optimizer,
            tokens, config, (10, 14), bf16=True,
        )
        torch.testing.assert_close(replay_loss, expected_loss, rtol=0, atol=0)
        for name, gradient in expected_gradients.items():
            torch.testing.assert_close(replay_gradients[name], gradient, rtol=0, atol=0, msg=name)
        for name, tensor in parallel.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0, msg=name)
        torch.testing.assert_close(
            parallel_optimizer.state_dict(), restored_optimizer.state_dict(), rtol=0, atol=0
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        dist.destroy_process_group()


def test_two_rank_full_model_ple_ep_training_and_restart():
    """Cover GDN, QSA, HC, PLE, EP, global losses, DCP, and AdamW replay."""
    if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip("Two CUDA devices and NCCL are required for full QwenAir EP training")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="qwenair-full-ep-") as root:
        checkpoint_dir = Path(root) / "sharded"
        checkpoint_dir.mkdir()
        mp.spawn(_worker, args=(port, str(checkpoint_dir)), nprocs=2, join=True)
