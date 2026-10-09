# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for QwenAir's opt-in packed Transformer Engine expert backend."""

from __future__ import annotations

import pytest
import torch

from megatron.core.models.qwenair import QwenAirTextConfig
from megatron.core.models.qwenair.layers import QwenAirExperts
from megatron.core.models.qwenair.moe_grouped import QwenAirDispatchedExpertBackend


def _config(**overrides) -> QwenAirTextConfig:
    values = dict(
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
    values.update(overrides)
    return QwenAirTextConfig(**values)


def test_expert_backend_config_is_explicit_and_validated():
    """The reference loop stays the default and unknown backends are rejected."""
    assert _config().moe_expert_backend == "loop"
    assert _config(moe_expert_backend="te_grouped").moe_expert_backend == "te_grouped"
    with pytest.raises(ValueError, match="moe_expert_backend"):
        _config(moe_expert_backend="unknown")


def test_distributed_training_cli_only_overrides_backend_when_explicit():
    """The training CLI preserves config defaults unless the override is present."""
    from examples.qwenair.train_distributed import load_config, parse_args

    default_args = parse_args([])
    assert default_args.moe_expert_backend is None
    assert load_config(default_args).moe_expert_backend == "loop"

    grouped_args = parse_args(["--moe-expert-backend", "te_grouped"])
    assert load_config(grouped_args).moe_expert_backend == "te_grouped"


def test_te_grouped_backend_requires_explicit_experimental_opt_in(monkeypatch):
    """A recipe cannot select the experimental TE layout without its env gate."""
    monkeypatch.delenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", raising=False)
    experts = QwenAirExperts(_config(), num_local_experts=4)
    with pytest.raises(RuntimeError, match="NVTE_GROUPED_LINEAR_SINGLE_PARAM=1"):
        QwenAirDispatchedExpertBackend(experts, "te_grouped")


def test_loop_backend_preserves_parameter_tree_and_handles_ragged_zero_counts():
    """The CPU fallback remains the checkpoint oracle, including empty experts."""
    torch.manual_seed(71)
    experts = QwenAirExperts(_config(), num_local_experts=4)
    reference = QwenAirExperts(_config(), num_local_experts=4)
    reference.load_state_dict(experts.state_dict(), strict=True)
    parameter_ids = {name: id(param) for name, param in experts.named_parameters()}
    state_keys = set(experts.state_dict())
    backend = QwenAirDispatchedExpertBackend(experts, "loop")

    counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    expert_indices = torch.tensor([0, 0, 2, 2, 2, 3], dtype=torch.int64)
    scores = torch.tensor([0.2, 0.8, 0.1, 0.6, 0.4, 0.9])
    hidden = torch.randn(6, 16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()

    actual = backend(hidden, counts, scores)
    expected = reference(reference_hidden, expert_indices.unsqueeze(-1), scores.unsqueeze(-1))
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(experts.gate_up_proj.grad, reference.gate_up_proj.grad)
    torch.testing.assert_close(experts.down_proj.grad, reference.down_proj.grad)

    assert set(experts.state_dict()) == state_keys == {"gate_up_proj", "down_proj"}
    assert {name: id(param) for name, param in experts.named_parameters()} == parameter_ids
    assert tuple(experts.gate_up_proj.shape) == (4, 16, 16)
    assert tuple(experts.down_proj.shape) == (4, 16, 8)

    experts.zero_grad(set_to_none=True)
    empty_hidden = torch.empty(0, 16, requires_grad=True)
    empty_output = backend(empty_hidden, torch.zeros(4, dtype=torch.int64), torch.empty(0))
    assert tuple(empty_output.shape) == (0, 16)
    empty_output.sum().backward()
    assert empty_hidden.grad is not None
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None
    assert torch.count_nonzero(experts.gate_up_proj.grad) == 0
    assert torch.count_nonzero(experts.down_proj.grad) == 0


@pytest.mark.parametrize("counts", ([2, 0, 3, 1], [0, 0, 0, 0]))
def test_te_grouped_backend_matches_bf16_forward_and_gradients(monkeypatch, counts):
    """Supported CUDA/TE systems match the loop for ragged and zero-sized groups."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TE grouped expert parity test")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")

    torch.manual_seed(73)
    reference = QwenAirExperts(_config(), num_local_experts=4)
    grouped = QwenAirExperts(_config(), num_local_experts=4)
    grouped.load_state_dict(reference.state_dict(), strict=True)
    setattr(grouped.gate_up_proj, "allreduce", False)
    setattr(grouped.down_proj, "allreduce", False)
    try:
        backend = QwenAirDispatchedExpertBackend(grouped, "te_grouped")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    device = torch.device("cuda", torch.cuda.current_device())
    reference = reference.to(device=device, dtype=torch.bfloat16)
    grouped = grouped.to(device=device, dtype=torch.bfloat16)
    ordinary_parameter_ids = {name: id(parameter) for name, parameter in grouped.named_parameters()}
    try:
        prepared = backend.prepare_after_module_apply()
    except RuntimeError as exc:
        if "unsupported on this GPU" in str(exc):
            pytest.skip(str(exc))
        raise
    assert prepared and backend.is_prepared

    assert set(grouped.state_dict()) == {"gate_up_proj", "down_proj"}
    assert set(dict(grouped.named_parameters())) == {"gate_up_proj", "down_proj"}
    assert tuple(grouped.gate_up_proj.shape) == tuple(reference.gate_up_proj.shape)
    assert tuple(grouped.down_proj.shape) == tuple(reference.down_proj.shape)
    grouped_parameter_ids = {name: id(parameter) for name, parameter in grouped.named_parameters()}
    assert grouped_parameter_ids.keys() == ordinary_parameter_ids.keys()
    assert all(
        grouped_parameter_ids[name] != ordinary_parameter_ids[name]
        for name in grouped_parameter_ids
    )
    assert grouped.gate_up_proj.allreduce is False
    assert grouped.down_proj.allreduce is False
    assert backend.prepare_after_module_apply()
    assert {
        name: id(parameter) for name, parameter in grouped.named_parameters()
    } == grouped_parameter_ids
    grouped.load_state_dict(reference.state_dict(), strict=True)
    ordinary_reload = QwenAirExperts(_config(), num_local_experts=4).to(
        device=device, dtype=torch.bfloat16
    )
    ordinary_reload.load_state_dict(grouped.state_dict(), strict=True)
    torch.testing.assert_close(ordinary_reload.gate_up_proj, reference.gate_up_proj)
    torch.testing.assert_close(ordinary_reload.down_proj, reference.down_proj)

    counts_tensor = torch.tensor(counts, dtype=torch.int64)
    num_tokens = sum(counts)
    hidden_data = torch.randn(num_tokens, 16, device=device, dtype=torch.bfloat16)
    scores = torch.rand(num_tokens, device=device, dtype=torch.float32)
    reference_hidden = hidden_data.detach().clone().requires_grad_()
    grouped_hidden = hidden_data.detach().clone().requires_grad_()

    expected = reference.forward_dispatched(reference_hidden, counts_tensor, scores)
    actual = backend(grouped_hidden, counts_tensor, scores)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=5e-3)

    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)
    actual.backward(grad_output)
    torch.testing.assert_close(
        grouped_hidden.grad.float(), reference_hidden.grad.float(), rtol=1e-2, atol=5e-3
    )
    for name in ("gate_up_proj", "down_proj"):
        expected_grad = getattr(reference, name).grad
        actual_grad = getattr(grouped, name).grad
        assert expected_grad is not None and actual_grad is not None
        torch.testing.assert_close(
            actual_grad.float(), expected_grad.float(), rtol=1e-2, atol=5e-3, msg=name
        )
