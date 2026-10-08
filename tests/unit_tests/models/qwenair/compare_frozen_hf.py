# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare QwenAir reference math against the frozen Qwen4-Exp HF source.

Run from the Megatron-LM checkout with its normal Python dependencies:

    python tests/unit_tests/models/qwenair/compare_frozen_hf.py \
        --hf-repo ../huggingface-modeling-code

The script extracts only selected definitions from the frozen Git object. It
removes integration decorators so their PyTorch fallbacks run without a full
Transformers installation; it does not copy or edit the upstream source.
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.models.qwenair import QwenAirTextConfig
from megatron.core.models.qwenair.layers import (
    QwenAirGatedDeltaNet,
    QwenAirGatedResidual,
    qwenair_rope,
)
from megatron.core.models.qwenair.ple import QwenAirPLE
from megatron.core.models.qwenair.qsa import QwenAirQSA, QwenAirQSAIndexer

HF_COMMIT = "2ff8a4b2752cb54ff8dedfd7408ac7e6b7d2ee40"
HF_SOURCE = "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
HF_DEFINITIONS = {
    "Qwen4ExpTextRMSNorm",
    "Qwen4ExpTextRMSNormGated",
    "apply_mask_to_padding_states",
    "causal_conv1d_fn",
    "l2norm",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "Qwen4ExpTextGatedDeltaNet",
    "rotate_half",
    "apply_rotary_pos_emb",
    "Qwen4ExpTextQSAIndexer",
    "repeat_kv",
    "eager_attention_forward",
    "Qwen4ExpTextAttention",
    "Qwen4ExpTextGatedResidual",
    "_splitmix64",
    "_build_layer_multipliers",
    "_is_prime",
    "_find_nth_prime_after",
    "Qwen4ExpTextNGramEmbedding",
    "Qwen4ExpTextPLELayer",
}
HF_CONSTANTS = {"_MASK64", "_SPLITMIX_GAMMA", "_SPLITMIX_M1", "_SPLITMIX_M2", "_PRIME_1"}


class _EagerAttentionRegistry:
    def get_interface(self, _name, fallback):
        return fallback


def _load_hf(repo: Path) -> dict:
    source = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{HF_COMMIT}:{HF_SOURCE}"], text=True, encoding="utf-8"
    )
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in HF_DEFINITIONS:
            for nested in ast.walk(node):
                if hasattr(nested, "decorator_list"):
                    nested.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in HF_CONSTANTS for target in node.targets
        ):
            selected.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "ACT2FN": {"silu": F.silu, "sigmoid": torch.sigmoid},
        "ALL_ATTENTION_FUNCTIONS": _EagerAttentionRegistry(),
    }
    exec(compile(module, f"{HF_COMMIT}:{HF_SOURCE}", "exec"), namespace)
    return namespace


def _tiny_config() -> QwenAirTextConfig:
    config = QwenAirTextConfig(
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
        linear_conv_kernel_dim=3,
        hc_count=4,
        hc_lowrank=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        indexer_n_heads=2,
        indexer_head_dim=4,
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
    )
    config.hidden_act = "silu"
    config._attn_implementation = "eager"
    return config


def compare(repo: Path) -> None:
    """Check strict parameter layouts and deterministic FP32 forward values."""
    hf = _load_hf(repo)
    config = _tiny_config()
    torch.manual_seed(114)
    specs = (
        ("HC", QwenAirGatedResidual, "Qwen4ExpTextGatedResidual", ()),
        ("PLE", QwenAirPLE, "Qwen4ExpTextPLELayer", (0, 0)),
        ("GDN", QwenAirGatedDeltaNet, "Qwen4ExpTextGatedDeltaNet", (0,)),
        ("QSA indexer", QwenAirQSAIndexer, "Qwen4ExpTextQSAIndexer", (1,)),
        ("QSA attention", QwenAirQSA, "Qwen4ExpTextAttention", (1,)),
    )
    length = 13
    positions = torch.arange(length).expand(2, -1)
    cos, sin = qwenair_rope(config, positions, torch.float32)
    visible = torch.ones(length, length, dtype=torch.bool).tril().unsqueeze(0).expand(2, -1, -1)
    for label, ours_class, hf_name, args in specs:
        ours = ours_class(config, *args)
        reference = hf[hf_name](config, *args)
        reference.load_state_dict(ours.state_dict(), strict=True)
        if label == "HC":
            x = torch.randn(2, 5, config.hc_count * config.hidden_size)
            for actual, expected in zip(ours(x), reference(x)):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        elif label == "PLE":
            x = torch.randn(2, 5, config.hc_count * config.hidden_size)
            ids = torch.tensor([[1, 2, 3, 4, 5], [3, 6, 7, 8, 9]])
            torch.testing.assert_close(ours(x, ids), reference(x, ids, None), rtol=1e-5, atol=1e-6)
        elif label == "GDN":
            x = torch.randn(2, 5, config.hidden_size)
            torch.testing.assert_close(ours(x), reference(x), rtol=1e-4, atol=1e-5)
        elif label == "QSA indexer":
            x = torch.randn(2, length, config.hidden_size)
            actual = ours(x, cos, sin, visible).token_mask.unsqueeze(1)
            expected = reference(x, (cos, sin), visible.unsqueeze(1), None)
            torch.testing.assert_close(actual, expected)
        else:
            x = torch.randn(2, length, config.hidden_size)
            additive = torch.where(
                visible.unsqueeze(1), torch.tensor(0.0), torch.tensor(torch.finfo(torch.float32).min)
            )
            actual = ours(x, cos, sin, visible)
            expected, _ = reference(x, (cos, sin), additive)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        print(f"PASS {label}: strict state_dict and FP32 forward parity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-repo", type=Path,
        default=Path(__file__).resolve().parents[5] / "huggingface-modeling-code",
    )
    compare(parser.parse_args().hf_repo)
