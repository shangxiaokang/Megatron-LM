# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run two QwenAir text reference training steps and verify checkpoint restart.

This is a small, single-rank functional gate, not a target-size pretraining job.
It exercises GDN, QSA, hyper-connections, PLE, MoE, BF16 autocast, and AdamW.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirTextConfig


def tiny_config(backend: str) -> QwenAirTextConfig:
    """Use the target block types with dimensions that fit on one device."""
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
        qsa_backend=backend,
    )


def training_step(
    model: QwenAirForCausalLM,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
    """Take an optimizer step and check every implemented module gets gradients."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(tokens, labels=tokens, output_router_logits=True)
    if output.loss is None or output.aux_loss is None or not torch.isfinite(output.loss):
        raise AssertionError("QwenAir text loss and router loss must be finite")
    output.loss.backward()
    parameters = dict(model.named_parameters())
    required = {
        "hc": "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "ple": "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        "gdn": "model.layers.0.linear_attn.in_proj_qkv.weight",
        "qsa": "model.layers.3.self_attn.q_proj.weight",
        "moe": "model.layers.3.mlp.experts.gate_up_proj",
        "head": "lm_head.weight",
    }
    norms = {}
    gradient_snapshots = {}
    for module, name in required.items():
        grad = parameters[name].grad
        if grad is None or not torch.isfinite(grad).all():
            raise AssertionError(f"{module} has no finite gradient: {name}")
        norms[module] = float(grad.float().norm().item())
        gradient_snapshots[module] = grad.detach().clone()
    indexer = parameters["model.layers.3.self_attn.indexer.index_qk_proj.weight"]
    if indexer.grad is not None:
        raise AssertionError("hard top-k indexer unexpectedly received an LM gradient")
    optimizer.step()
    return float(output.loss.detach().item()), norms, gradient_snapshots


def main() -> None:
    """Check BF16 training and replay the second step after checkpoint reload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("dense", "te_reference"), default="dense")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but is unavailable")
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = tiny_config(args.backend)
    batches = [torch.randint(1, config.vocab_size, (2, 16), device=device) for _ in range(2)]
    model = QwenAirForCausalLM(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    if args.checkpoint_dir is None:
        temporary = TemporaryDirectory(prefix="qwenair-reference-")
        checkpoint_dir = Path(temporary.name)
    else:
        temporary = None
        checkpoint_dir = args.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        first_loss, first_norms, _ = training_step(model, optimizer, batches[0], device)
        checkpoint = checkpoint_dir / "step_1.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        second_loss, second_norms, second_gradients = training_step(model, optimizer, batches[1], device)

        restored = QwenAirForCausalLM(config).to(device).train()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
        # Keep AdamW's scalar step counters on CPU, as in the live optimizer.
        # load_state_dict moves moment tensors to their parameter devices.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(saved["model"], strict=True)
        restored_optimizer.load_state_dict(saved["optimizer"])
        replay_loss, replay_norms, replay_gradients = training_step(
            restored, restored_optimizer, batches[1], device
        )
        torch.testing.assert_close(torch.tensor(second_loss), torch.tensor(replay_loss), rtol=0, atol=0)
        for module, gradient in second_gradients.items():
            torch.testing.assert_close(gradient, replay_gradients[module], rtol=0, atol=0, msg=module)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0, msg=key)
        torch.testing.assert_close(
            optimizer.state_dict(), restored_optimizer.state_dict(), rtol=0, atol=0
        )
        if not all(value > 0 for value in first_norms.values()):
            raise AssertionError("a module had zero gradient in the first step")
        print(json.dumps({
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "backend": args.backend,
            "torch": torch.__version__,
            "loss": [first_loss, second_loss],
            "grad_norms": [first_norms, second_norms],
            "replay_grad_norms": replay_norms,
            "checkpoint": str(checkpoint),
            "restart_exact": True,
        }, sort_keys=True))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
