# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir text topology, selection, and differentiable training tests."""

import math
from copy import deepcopy

import pytest
import torch

from megatron.core.models.qwenair import QwenAirForCausalLM, QwenAirTextConfig
from megatron.core.models.qwenair.layers import qwenair_global_router_loss
from megatron.core.models.qwenair.model import _validate_te_qsa_token_mask


def tiny_config(**overrides):
    """Build a tractable QwenAir shape with both GDN and QSA layers."""
    values = dict(
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
        indexer_budget=4,
        indexer_compress_ratio=2,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 0),
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=3,
        mtp_num_hidden_layers=1,
    )
    values.update(overrides)
    return QwenAirTextConfig(**values)


def test_target_config_preserves_48_layer_schedule():
    """The supplied target config maps to 36 GDN and 12 QSA layers."""
    config = QwenAirTextConfig.from_hf_dict(
        {
            "text_config": {
                "num_hidden_layers": 48,
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
                "ple_layer_ids": [2],
                "mtp_num_hidden_layers": 1,
                "mamba_ssm_dtype": "float32",
                "split_ngram_parts": 128,
            }
        }
    )
    assert config.layer_types.count("linear_attention") == 36
    assert config.layer_types.count("qwen_sparse_attention") == 12
    assert config.layer_types[3::4] == ["qwen_sparse_attention"] * 12
    assert config.ple_layer_ids == [2]
    assert config.rotary_dim == 64
    assert config.mamba_ssm_dtype == "float32"
    assert config.split_ngram_parts == 128


def test_target_config_rejects_non_fp32_gdn_state_contract():
    """The canonical GDN state contract cannot be silently weakened."""
    with pytest.raises(ValueError, match="mamba_ssm_dtype=float32"):
        tiny_config(mamba_ssm_dtype="bfloat16")


def test_qsa_selection_is_causal_and_keeps_incomplete_tail():
    """Every query has its own top-k blocks and up to ratio-1 tail tokens."""
    config = tiny_config()
    indexer = QwenAirForCausalLM(config).model.layers[1].self_attn.indexer
    hidden = torch.randn(1, 9, config.hidden_size)
    from megatron.core.models.qwenair.layers import qwenair_rope

    cos, sin = qwenair_rope(config, torch.arange(9).unsqueeze(0), hidden.dtype)
    visible = torch.ones(9, 9, dtype=torch.bool).tril().unsqueeze(0)
    selection = indexer(hidden, cos, sin, visible)
    assert selection.token_mask.shape == (1, 9, 9)
    assert not (selection.token_mask & ~visible).any()
    assert selection.token_mask[0, 0, 0]
    assert selection.token_mask[0, 8].sum() == config.indexer_budget + 1
    assert selection.tail_tokens[0, 8, 0] == 8
    assert (selection.block_starts[0, 8] >= 0).sum() == 2


def test_tiny_text_training_has_main_gradients_and_no_indexer_lm_gradient():
    """One CE step trains HC, PLE, GDN, QSA and experts; hard top-k isolates indexer."""
    torch.manual_seed(6)
    model = QwenAirForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [3, 7, 8, 9, 10, 11]])
    labels = input_ids.clone()
    result = model(input_ids, labels=labels, output_router_logits=True)
    assert result.logits.shape == (2, 6, 64)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    parameters = dict(model.named_parameters())
    for name in (
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.ple.ple_embedding.ngram_embedding.weight",
        "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "lm_head.weight",
    ):
        assert parameters[name].grad is not None, name
        assert torch.isfinite(parameters[name].grad).all(), name
    assert parameters["model.layers.1.self_attn.indexer.index_qk_proj.weight"].grad is None
    old = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(old, model.lm_head.weight)


def test_pre_shifted_megatron_labels_match_hf_internal_shift():
    """Accept GPTDataset next-token labels without shifting them a second time."""
    torch.manual_seed(62)
    model = QwenAirForCausalLM(tiny_config())
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    hf_output = model(tokens, labels=tokens)
    pre_shifted_labels = torch.cat((tokens[:, 1:], tokens.new_full((1, 1), -100)), dim=1)
    megatron_output = model(tokens, labels=pre_shifted_labels, labels_are_shifted=True)
    torch.testing.assert_close(megatron_output.logits, hf_output.logits)
    torch.testing.assert_close(megatron_output.loss, hf_output.loss)


def test_pp1_schedule_input_hook_is_fail_closed():
    """Accept MCore's PP=1 sentinel and reject a real pipeline activation."""
    model = QwenAirForCausalLM(tiny_config())
    model.set_input_tensor(None)
    model.set_input_tensor([None])
    model.set_input_tensor((None,))
    with pytest.raises(NotImplementedError, match="PP=1"):
        model.set_input_tensor(torch.zeros(1, 2, model.config.hidden_size))


@pytest.mark.parametrize(
    "token_mask",
    (
        torch.tensor([[True, True, True]]),
        torch.tensor([[True, True, False, False]]),
        torch.tensor([[True, False, False], [True, True, False]]),
    ),
)
def test_te_qsa_accepts_unpadded_or_right_padded_token_masks(token_mask):
    """A masked suffix cannot affect earlier causal outputs."""
    _validate_te_qsa_token_mask(token_mask)


@pytest.mark.parametrize(
    "token_mask",
    (
        torch.tensor([[False, True, True]]),
        torch.tensor([[True, False, True]]),
        torch.tensor([[False, False, False]]),
    ),
)
def test_te_qsa_rejects_left_gapped_or_empty_token_masks(token_mask):
    """TE QSA must reject masks that change valid-prefix attention."""
    with pytest.raises(NotImplementedError, match="right-padded suffix|valid token"):
        _validate_te_qsa_token_mask(token_mask)


def test_right_padding_matches_unpadded_prefix_objective_and_gradients():
    """A causal right-padded suffix is inert when every objective keeps its mask."""
    torch.manual_seed(63)
    model = QwenAirForCausalLM(tiny_config())
    tokens = torch.tensor([[1, 2, 3, 4, 5, 3, 3, 3]])
    token_mask = torch.tensor([[True, True, True, True, True, False, False, False]])
    labels = torch.tensor([[2, 3, 4, 5, -100, -100, -100, -100]])

    padded = model(
        tokens,
        attention_mask=token_mask,
        labels=labels,
        labels_are_shifted=True,
        output_router_logits=True,
    )
    padded.loss.backward()
    gradient_names = (
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate.weight",
    )
    padded_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if name in gradient_names
    }
    model.zero_grad(set_to_none=True)

    unpadded = model(
        tokens[:, :5],
        labels=labels[:, :5],
        labels_are_shifted=True,
        output_router_logits=True,
    )
    unpadded.loss.backward()

    torch.testing.assert_close(padded.logits[:, :5], unpadded.logits)
    torch.testing.assert_close(padded.loss, unpadded.loss)
    torch.testing.assert_close(padded.aux_loss, unpadded.aux_loss)
    parameters = dict(model.named_parameters())
    assert padded_gradients.keys() == set(gradient_names)
    for name, padded_gradient in padded_gradients.items():
        torch.testing.assert_close(padded_gradient, parameters[name].grad)


def test_external_visual_embeddings_keep_original_ple_ids_and_gradients():
    """Visual scatter enters the text stream without changing PLE token history."""
    import pytest

    torch.manual_seed(61)
    model = QwenAirForCausalLM(tiny_config())
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    embeddings = model.model.embed_tokens(tokens).detach()
    positions = torch.arange(tokens.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    direct = model(tokens, position_ids=positions, labels=tokens)
    external = model(
        None, position_ids=positions, labels=tokens, inputs_embeds=embeddings, ple_input_ids=tokens
    )
    torch.testing.assert_close(external.logits, direct.logits)
    torch.testing.assert_close(external.loss, direct.loss)

    visual_patch = torch.randn(1, 1, model.config.hidden_size, requires_grad=True)
    mixed = torch.cat((embeddings[:, :2], visual_patch, embeddings[:, 3:]), dim=1)
    visual = model(
        None, position_ids=positions, labels=tokens, inputs_embeds=mixed, ple_input_ids=tokens
    )
    visual.loss.backward()
    assert visual_patch.grad is not None and torch.count_nonzero(visual_patch.grad)
    with pytest.raises(ValueError, match="exactly one"):
        model(tokens, inputs_embeds=embeddings)
    with pytest.raises(ValueError, match="ple_input_ids"):
        model(None, inputs_embeds=embeddings)


def test_tiny_text_training_with_cuda_bf16_autocast():
    """Routed expert accumulation keeps the residual dtype under BF16 autocast."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 training smoke test")
    model = QwenAirForCausalLM(tiny_config()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(tokens, labels=tokens, output_router_logits=True)
            assert torch.isfinite(output.loss)
            output.loss.backward()
            expert_grad = model.model.layers[0].mlp.experts.gate_up_proj.grad
            assert expert_grad is not None and torch.isfinite(expert_grad).all()
            optimizer.step()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_qwenair_fla_gdn_matches_token_reference_with_cuda_bf16(monkeypatch: pytest.MonkeyPatch):
    """The training FLA path preserves QwenAir GDN outputs and gradients."""
    from megatron.core.models.qwenair import layers as qwenair_layers

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 FLA comparison")
    if not qwenair_layers.HAVE_FLA or qwenair_layers.chunk_gated_delta_rule is None:
        pytest.skip("The FLA gated-delta kernel is unavailable")

    torch.manual_seed(67)
    config = tiny_config(num_hidden_layers=1, layer_types=["linear_attention"], ple_layer_ids=[])
    kernel_layer = qwenair_layers.QwenAirGatedDeltaNet(config, layer_idx=0).cuda()
    reference_layer = deepcopy(kernel_layer)
    kernel_input = torch.randn(2, 13, config.hidden_size, device="cuda", requires_grad=True)
    reference_input = kernel_input.detach().clone().requires_grad_(True)
    token_mask = torch.tensor(
        [[True] * 13, [True] * 9 + [False] * 4],
        device="cuda",
    )

    kernel_calls = 0
    kernel = qwenair_layers.chunk_gated_delta_rule

    def counted_kernel(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return kernel(*args, **kwargs)

    monkeypatch.setattr(qwenair_layers, "chunk_gated_delta_rule", counted_kernel)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kernel_output = kernel_layer(kernel_input, token_mask=token_mask)
    assert kernel_calls == 1
    monkeypatch.setattr(qwenair_layers, "HAVE_FLA", False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_output = reference_layer(reference_input, token_mask=token_mask)

    def assert_relative_l2(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
        difference = (actual.float() - expected.float()).norm()
        relative_l2 = difference / expected.float().norm().clamp_min(1e-12)
        assert relative_l2 < 8e-2, f"{name} relative L2 error is {relative_l2.item():.6f}"

    torch.testing.assert_close(kernel_output, reference_output, atol=5e-2, rtol=5e-2)
    assert_relative_l2(kernel_output, reference_output, "output")
    grad_generator = torch.Generator(device="cuda").manual_seed(73)
    grad_output = torch.randn(
        kernel_output.shape,
        device=kernel_output.device,
        dtype=kernel_output.dtype,
        generator=grad_generator,
    ) / math.sqrt(kernel_output.numel())
    kernel_output.backward(grad_output)
    reference_output.backward(grad_output)
    torch.testing.assert_close(kernel_input.grad, reference_input.grad, atol=6e-2, rtol=6e-2)
    assert_relative_l2(kernel_input.grad, reference_input.grad, "input gradient")
    reference_parameters = dict(reference_layer.named_parameters())
    for name, parameter in kernel_layer.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        reference_gradient = reference_parameters[name].grad
        assert reference_gradient is not None and torch.isfinite(reference_gradient).all(), name
        torch.testing.assert_close(parameter.grad, reference_gradient, atol=8e-2, rtol=8e-2)
        assert_relative_l2(parameter.grad, reference_gradient, f"{name} gradient")


def test_required_fused_gdn_fails_instead_of_silently_using_token_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """A target run cannot accidentally use the reference token recurrence."""
    from megatron.core.models.qwenair import layers as qwenair_layers

    config = tiny_config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[],
        require_fused_gdn=True,
    )
    layer = qwenair_layers.QwenAirGatedDeltaNet(config, layer_idx=0)
    monkeypatch.setattr(qwenair_layers, "HAVE_FLA", False)

    with pytest.raises(RuntimeError, match="requires the FLA chunk gated-delta kernel"):
        layer(torch.randn(1, 3, config.hidden_size))


def test_cross_layer_auxiliary_uses_all_router_logits():
    """The global load balancing loss aggregates counts before normalization."""
    logits = (
        torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True),
    )
    loss = qwenair_global_router_loss(logits, num_experts=2, top_k=1)
    probabilities = torch.softmax(torch.cat(logits, dim=0), dim=-1)
    selected = probabilities.argmax(dim=-1)
    counts = torch.bincount(selected, minlength=2).float() / 4
    expected = 2 * (counts * probabilities.mean(dim=0)).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits[0].grad is not None and logits[1].grad is not None


def test_hyper_connection_zero_projection_matches_closed_form():
    """Four-stream read and injection obey the frozen HF equations."""
    from megatron.core.models.qwenair.layers import QwenAirGatedResidual, inject_hyper_output

    config = tiny_config()
    cell = QwenAirGatedResidual(config)
    with torch.no_grad():
        cell.input_mix_weight_down.weight.zero_()
        cell.input_mix_weight_up.weight.zero_()
        cell.block_inject_weight.weight.zero_()
    state = torch.randn(2, 3, config.hc_count * config.hidden_size, requires_grad=True)
    read, original, injection = cell(state)
    normalized = cell.hc_norm(state).unflatten(-1, (config.hc_count, config.hidden_size))
    torch.testing.assert_close(read, 0.5 * normalized.mean(dim=-2))
    torch.testing.assert_close(injection, torch.ones_like(injection))
    block_output = torch.randn_like(read)
    merged = inject_hyper_output(original, block_output, injection)
    torch.testing.assert_close(merged, state + block_output.repeat(1, 1, config.hc_count))
    merged.square().mean().backward()
    assert state.grad is not None


def test_qsa_left_padding_and_short_context_keep_all_visible_tokens():
    """Under budget, QSA selection equals the masked causal visibility."""
    from megatron.core.models.qwenair.layers import qwenair_rope
    from megatron.core.models.qwenair.qsa import QwenAirQSAIndexer

    config = tiny_config(indexer_budget=16)
    indexer = QwenAirQSAIndexer(config, 0)
    hidden = torch.randn(1, 7, config.hidden_size)
    cos, sin = qwenair_rope(config, torch.arange(7).unsqueeze(0), hidden.dtype)
    valid = torch.tensor([[False, False, True, True, True, True, True]])
    visible = torch.ones(7, 7, dtype=torch.bool).tril().unsqueeze(0)
    visible &= valid[:, :, None] & valid[:, None, :]
    selected = indexer(hidden, cos, sin, visible)
    torch.testing.assert_close(selected.token_mask, visible)


def test_chunked_unpadded_qsa_indexer_matches_dense_selection():
    """Streaming query chunks retain the frozen per-token block and tail choices."""
    from megatron.core.models.qwenair.layers import qwenair_rope
    from megatron.core.models.qwenair.qsa import QwenAirQSAIndexer

    torch.manual_seed(43)
    config = tiny_config(indexer_budget=8, indexer_compress_ratio=4)
    indexer = QwenAirQSAIndexer(config, 1)
    hidden = torch.randn(2, 27, config.hidden_size)
    positions = torch.arange(27).expand(2, -1)
    cos, sin = qwenair_rope(config, positions, torch.float32)
    visible = torch.ones(27, 27, dtype=torch.bool).tril().expand(2, -1, -1)
    dense = indexer(hidden, cos, sin, visible)
    chunked = indexer(hidden, cos, sin, None)
    assert chunked.token_mask is None
    torch.testing.assert_close(chunked.tail_tokens, dense.tail_tokens)
    torch.testing.assert_close(chunked.block_starts, dense.block_starts)

    # All-zero scores force top-k ties at every completed-block boundary.
    for length in (1, 3, 4, 7, 8, 15, 16, 17, 31):
        tied_hidden = torch.zeros(2, length, config.hidden_size)
        tied_positions = torch.arange(length).expand(2, -1)
        tied_cos, tied_sin = qwenair_rope(config, tied_positions, torch.float32)
        tied_visible = torch.ones(length, length, dtype=torch.bool).tril().expand(2, -1, -1)
        dense_ties = indexer(tied_hidden, tied_cos, tied_sin, tied_visible)
        chunked_ties = indexer(tied_hidden, tied_cos, tied_sin, None)
        torch.testing.assert_close(chunked_ties.block_starts, dense_ties.block_starts)
        torch.testing.assert_close(chunked_ties.tail_tokens, dense_ties.tail_tokens)


@pytest.mark.parametrize("backend", ["te_reference", "te_indexed_sdpa"])
def test_te_reference_can_exceed_dense_mask_limit_without_square_selection(backend):
    """An unpacked TE sequence uses only selected block indices and tail tokens."""
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.models.qwenair.layers import qwenair_rope
    from megatron.core.models.qwenair.qsa import QwenAirQSA

    torch.manual_seed(44)
    config = tiny_config(
        indexer_budget=8,
        indexer_compress_ratio=4,
        qsa_backend=backend,
        max_reference_sequence_length=5,
    )
    model = QwenAirForCausalLM(config)
    tokens = torch.arange(1, 14).unsqueeze(0)
    output = model(tokens, labels=tokens)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    qsa = model.model.layers[1].self_attn
    hidden = torch.randn(1, 13, config.hidden_size)
    cos, sin = qwenair_rope(config, torch.arange(13).unsqueeze(0), torch.float32)
    chunked = qsa.indexer(hidden, cos, sin, None)
    assert chunked.token_mask is None
    assert chunked.block_starts.shape[:2] == (1, 13)
    with pytest.raises(NotImplementedError, match="unpadded"):
        qsa(hidden, cos, sin, torch.zeros(1, 13, 13, dtype=torch.bool))


def test_left_padding_cannot_change_valid_ple_outputs_and_four_axis_rope():
    """Masked PLE history uses EOS, and Qwen's text axis stays out of MRoPE."""
    from megatron.core.models.qwenair.layers import qwenair_rope

    config = tiny_config()
    model = QwenAirForCausalLM(config).eval()
    first = torch.tensor([[11, 12, 3, 4, 5, 6]])
    second = torch.tensor([[20, 21, 3, 4, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        first_logits = model(first, attention_mask=mask).logits
        second_logits = model(second, attention_mask=mask).logits
    torch.testing.assert_close(first_logits[:, 2:], second_logits[:, 2:])
    text_positions = torch.arange(6).unsqueeze(0)
    four_axes = text_positions.unsqueeze(0).expand(4, -1, -1).clone()
    four_axes[0] += 20
    base_cos, base_sin = qwenair_rope(config, text_positions, torch.float32)
    four_cos, four_sin = qwenair_rope(config, four_axes, torch.float32)
    torch.testing.assert_close((base_cos, base_sin), (four_cos, four_sin))


@pytest.mark.parametrize("backend", ["te_reference", "te_indexed_sdpa"])
def test_qsa_te_reference_matches_dense_outputs_and_gradients(backend):
    """A per-token four-key-block TE selection preserves dense QSA training math."""
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.models.qwenair.layers import qwenair_rope
    from megatron.core.models.qwenair.qsa import QwenAirQSA

    torch.manual_seed(13)
    dense_config = tiny_config(indexer_compress_ratio=4, indexer_budget=8)
    te_config = tiny_config(indexer_compress_ratio=4, indexer_budget=8, qsa_backend=backend)
    dense = QwenAirQSA(dense_config, 1)
    te = QwenAirQSA(te_config, 1)
    te.load_state_dict(dense.state_dict(), strict=True)
    dense_input = torch.randn(2, 15, dense_config.hidden_size, requires_grad=True)
    te_input = dense_input.detach().clone().requires_grad_()
    positions = torch.arange(15).expand(2, -1)
    cos, sin = qwenair_rope(dense_config, positions, torch.float32)
    visible = torch.ones(15, 15, dtype=torch.bool).tril().unsqueeze(0).expand(2, -1, -1)
    dense_output = dense(dense_input, cos, sin, visible)
    te_output = te(te_input, cos, sin, visible)
    torch.testing.assert_close(te_output, dense_output, rtol=2e-5, atol=2e-6)
    dense_output.square().sum().backward()
    te_output.square().sum().backward()
    torch.testing.assert_close(te_input.grad, dense_input.grad, rtol=5e-5, atol=5e-6)
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
        dense_grad = dict(dense.named_parameters())[name].grad
        te_grad = dict(te.named_parameters())[name].grad
        torch.testing.assert_close(te_grad, dense_grad, rtol=5e-5, atol=5e-6)

    padded = visible.clone()
    padded[0, :, 0] = False
    with pytest.raises(NotImplementedError, match="unpadded"):
        te(te_input, cos, sin, padded)


@pytest.mark.parametrize("backend", ["te_reference", "te_indexed_sdpa", "te_triton"])
def test_qsa_te_reference_matches_dense_with_cuda_bf16_autocast(backend):
    """FP32 score accumulation survives CUDA autocast in both QSA paths."""
    pytest.importorskip("transformer_engine.pytorch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 autocast comparison")
    from megatron.core.models.qwenair.layers import qwenair_rope
    from megatron.core.models.qwenair.qsa import QwenAirQSA

    torch.manual_seed(24)
    dense = QwenAirQSA(tiny_config(indexer_compress_ratio=4, indexer_budget=8), 1).cuda().bfloat16()
    te = (
        QwenAirQSA(tiny_config(indexer_compress_ratio=4, indexer_budget=8, qsa_backend=backend), 1)
        .cuda()
        .bfloat16()
    )
    te.load_state_dict(dense.state_dict(), strict=True)
    dense_input = torch.randn(1, 13, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    te_input = dense_input.detach().clone().requires_grad_()
    positions = torch.arange(13, device="cuda").unsqueeze(0)
    cos, sin = qwenair_rope(tiny_config(), positions, torch.bfloat16)
    visible = torch.ones(13, 13, dtype=torch.bool, device="cuda").tril().unsqueeze(0)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dense_output = dense(dense_input, cos, sin, visible)
            te_output = te(te_input, cos, sin, visible)
        torch.testing.assert_close(te_output, dense_output, rtol=0.02, atol=0.002)
        dense_output.float().square().sum().backward()
        te_output.float().square().sum().backward()
        torch.testing.assert_close(te_input.grad, dense_input.grad, rtol=0.03, atol=0.003)
        torch.backends.cuda.matmul.allow_tf32 = True
        # BF16 smoke remains available under a framework-wide TF32 default.
        dense(dense_input, cos, sin, visible)
        strict_fp32 = QwenAirQSA(tiny_config(indexer_compress_ratio=4, indexer_budget=8), 1).cuda()
        with pytest.raises(NotImplementedError, match="allow_tf32=False"):
            strict_fp32(dense_input.float(), cos.float(), sin.float(), visible)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("backend", ["te_reference", "te_indexed_sdpa", "te_triton"])
def test_full_text_te_reference_trains_with_fp32_weights_and_bf16_autocast(backend):
    """RoPE-promoted FP32 Q/K and BF16 V keep exact TE input dtype handling."""
    pytest.importorskip("transformer_engine.pytorch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the mixed precision TE integration")
    torch.manual_seed(25)
    dense = QwenAirForCausalLM(tiny_config(indexer_compress_ratio=4, indexer_budget=8)).cuda()
    te = QwenAirForCausalLM(
        tiny_config(indexer_compress_ratio=4, indexer_budget=8, qsa_backend=backend)
    ).cuda()
    te.load_state_dict(dense.state_dict(), strict=True)
    tokens = torch.arange(1, 14, device="cuda").unsqueeze(0)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dense_output = dense(tokens, labels=tokens)
            te_output = te(tokens, labels=tokens)
        torch.testing.assert_close(te_output.logits, dense_output.logits, rtol=0.02, atol=0.01)
        dense_output.loss.backward()
        te_output.loss.backward()
        qsa_grad = te.model.layers[1].self_attn.q_proj.weight.grad
        assert qsa_grad is not None and torch.isfinite(qsa_grad).all()
        torch.testing.assert_close(
            qsa_grad, dense.model.layers[1].self_attn.q_proj.weight.grad, rtol=0.05, atol=0.002
        )
        optimizer = torch.optim.AdamW(te.parameters(), lr=0.001)
        optimizer.step()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_logical_state_dict_roundtrip_and_mtp_failure():
    """Single-rank keys reload exactly while unapproved MTP stays closed."""
    import pytest

    config = tiny_config()
    source = QwenAirForCausalLM(config).eval()
    target = QwenAirForCausalLM(config).eval()
    target.load_state_dict(source.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(source(tokens).logits, target(tokens).logits)
    with pytest.raises(NotImplementedError, match="MTP training"):
        source(tokens, enable_mtp=True)


def test_target_ple_cannot_allocate_on_single_rank():
    """A bare target-size construction fails before allocating the 95 GiB PLE table."""
    import pytest

    with pytest.raises(ValueError, match="distributed table"):
        QwenAirForCausalLM(QwenAirTextConfig())


def test_reference_rejects_long_context_before_allocating_quadratic_mask():
    """A target context needs the future sparse selector, not an S-squared mask."""
    import pytest

    model = QwenAirForCausalLM(tiny_config(max_reference_sequence_length=5))
    with pytest.raises(NotImplementedError, match="long-context training"):
        model(torch.tensor([[1, 2, 3, 4, 5, 6]]))


def test_target_ple_hash_matches_frozen_static_oracle_without_allocating_table():
    """Match model-info/p0/oracles/static-oracles.json for the real target shape."""
    import hashlib

    from megatron.core.models.qwenair.ple import qwenair_ngram_indices, qwenair_ngram_metadata

    config = QwenAirTextConfig()
    multipliers, sizes, offsets, rows = qwenair_ngram_metadata(config)
    assert multipliers.tolist() == [23703573157769, 20109073645365, 8052911324071]
    assert sizes.tolist() == [
        20000003,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000093,
        20000107,
        20000147,
        20000153,
        20000159,
        20000161,
        20000171,
    ]
    assert offsets.tolist() == [
        0,
        20000003,
        40000026,
        60000059,
        80000106,
        100000165,
        120000228,
        140000297,
        160000374,
        180000455,
        200000548,
        220000655,
        240000802,
        260000955,
        280001114,
        300001275,
    ]
    assert rows == 320001536
    input_ids = torch.tensor([[1, 2, 3, 248044, 4, 5], [248044, 7, 248044, 8, 9, 10]])
    indices = qwenair_ngram_indices(config, input_ids)
    assert indices.shape == (2, 6, 16)
    packed = b"".join(
        value.to_bytes(8, "little", signed=True) for value in indices.flatten().tolist()
    )
    assert (
        hashlib.sha256(packed).hexdigest()
        == "703516065311838305f34db5071c69e18fd29f9a5e972d5576c29c305fae7235"
    )


def test_target_text_parameter_count_matches_static_oracle_on_meta():
    """Build all 48 logical layers without materializing the 95 GiB PLE table."""
    config = QwenAirTextConfig(
        max_single_rank_ple_elements=10**12, max_single_rank_parameters=10**12
    )
    with torch.device("meta"):
        model = QwenAirForCausalLM(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 176943899520
    assert len(tuple(model.buffers())) == 3
