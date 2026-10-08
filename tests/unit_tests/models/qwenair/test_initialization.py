# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir topology-independent parameter initialization tests."""

from __future__ import annotations

import torch

from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirTextConfig
from megatron.core.models.qwenair.initialization import initialize_qwenair_sharded_normal_
from megatron.core.models.qwenair.layers import QwenAirExperts
from megatron.core.models.qwenair.ple import QwenAirNGramEmbedding


def _config(**overrides) -> QwenAirTextConfig:
    values = dict(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        hc_count=2,
        hc_lowrank=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        partial_rotary_factor=0.5,
        mrope_section=(1, 0, 0),
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ngram_size=3,
        heads_per_ngram=1,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=8,
        seed=20261008,
    )
    values.update(overrides)
    return QwenAirTextConfig(**values)


def _initialize_slice(
    size: int, start: int, logical_numel: int, config: QwenAirTextConfig
) -> torch.Tensor:
    tensor = torch.empty(size)
    initialize_qwenair_sharded_normal_(
        tensor,
        global_element_start=start,
        logical_numel=logical_numel,
        base_seed=config.seed,
        namespace="test.logical.parameter",
        layer_idx=3,
        std=config.initializer_range,
        chunk_elements=8,
    )
    return tensor


def test_logical_tensor_initialization_is_independent_of_shard_boundaries():
    """Fixed logical chunks give full and unevenly sharded tensors identical values."""
    config = _config()
    logical_numel = 37
    whole = _initialize_slice(logical_numel, 0, logical_numel, config)
    shards = torch.cat(
        (
            _initialize_slice(5, 0, logical_numel, config),
            _initialize_slice(18, 5, logical_numel, config),
            _initialize_slice(14, 23, logical_numel, config),
        )
    )

    assert torch.equal(shards, whole)
    assert torch.equal(_initialize_slice(logical_numel, 0, logical_numel, config), whole)


def test_expert_initialization_uses_global_expert_identity_without_global_rng():
    """EP shards concatenate to EP=1 while construction leaves the caller RNG intact."""
    config = _config(ple_layer_ids=[])
    torch.manual_seed(19)
    state_before = torch.random.get_rng_state().clone()

    full = QwenAirExperts(config, layer_idx=0)
    first = QwenAirExperts(
        config, num_local_experts=2, layer_idx=0, first_global_expert=0
    )
    second = QwenAirExperts(
        config, num_local_experts=2, layer_idx=0, first_global_expert=2
    )

    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert torch.equal(
        torch.cat((first.gate_up_proj, second.gate_up_proj)), full.gate_up_proj
    )
    assert torch.equal(torch.cat((first.down_proj, second.down_proj)), full.down_proj)
    assert not torch.equal(full.gate_up_proj[0], full.gate_up_proj[2])


def test_model_initializer_preserves_logical_ple_initialization():
    """The model-wide embedding pass must not overwrite the PLE shard initializer."""
    config = _config()
    torch.manual_seed(23)
    state_before = torch.random.get_rng_state().clone()
    QwenAirNGramEmbedding(config, layer_idx=0, ple_layer_index=0)
    assert torch.equal(torch.random.get_rng_state(), state_before)

    model = QwenAirForCausalLM(config)
    table = model.model.layers[0].ple.ple_embedding
    expected = torch.empty_like(table.ngram_embedding.weight)
    initialize_qwenair_sharded_normal_(
        expected,
        global_element_start=table.shard_start * expected.shape[1],
        logical_numel=table.padded_rows * expected.shape[1],
        base_seed=config.seed,
        namespace="qwenair.ple.table",
        layer_idx=0,
        std=config.initializer_range,
    )

    assert torch.equal(table.ngram_embedding.weight, expected)
