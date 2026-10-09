# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration for the QwenAir text training reference.

The reference keeps the Hugging Face checkpoint geometry and parameter names.
Its single-rank implementation is deliberately bounded: a target-size model
must use a distributed implementation for PLE, experts, and attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping


@dataclass
class QwenAirTextConfig:
    """Qwen4-Exp text architecture and single-rank training limits."""

    vocab_size: int = 248320
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    eos_token_id: int = 248044
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    # The canonical HF config requires FP32 temporal/cache state semantics.
    # Fused training kernels may internally store chunk-boundary tensors in the
    # activation dtype while accumulating the recurrence in FP32.
    mamba_ssm_dtype: str = "float32"
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    num_experts: int = 512
    num_experts_per_tok: int = 10
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001
    # The TE backend is opt-in until its ragged single-weight path is validated
    # on the target training platform.  The loop remains the correctness oracle.
    moe_expert_backend: str = "loop"
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    # Physical HF checkpoint shard count. It does not change runtime PLE
    # geometry, but retaining it prevents config round-trips from drifting.
    split_ngram_parts: int = 128
    seed: int = 1234
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    output_gate_type: str = "sigmoid"
    qsa_backend: str = "dense"
    # Runtime safety switch used by target-size recipes. Small CPU/reference
    # tests may use the token recurrence, while full training must fail rather
    # than silently select that memory- and launch-bound fallback.
    require_fused_gdn: bool = False
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    mtp_num_hidden_layers: int = 0
    # Resource limit, not part of the model's checkpoint contract.  It stops a
    # lone process from accidentally allocating the ~95 GiB target PLE table.
    max_single_rank_ple_elements: int = 50_000_000
    max_single_rank_parameters: int = 100_000_000
    max_reference_sequence_length: int = 2048
    # MCore DDP must not divide gradients a second time after the QwenAir
    # trainer has normalized CE by the WORLD valid-token count and averaged
    # router auxiliary loss across EDP. Distributed training entry points set
    # this to True. It remains False for the standalone single-rank reference.
    calculate_per_token_loss: bool = False
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        """Normalize layer names and reject unsupported training layouts."""
        if not self.layer_types:
            if self.full_attention_interval <= 0:
                raise ValueError("full_attention_interval must be positive")
            self.layer_types = [
                (
                    "linear_attention"
                    if (i + 1) % self.full_attention_interval
                    else "qwen_sparse_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "qwen_sparse_attention" if kind == "full_attention" else kind
            for kind in self.layer_types
        ]
        self.ple_layer_ids = sorted(set(self.ple_layer_ids))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        self.validate()

    @classmethod
    def from_hf_dict(cls, config: Mapping[str, Any]) -> QwenAirTextConfig:
        """Read a flat text config or the ``text_config`` of a Qwen4-Exp VLM."""
        text = config.get("text_config", config)
        if not isinstance(text, Mapping):
            raise TypeError("text_config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        kwargs = {name: text[name] for name in allowed if name in text}
        rope = text.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default" or not rope.get("mrope_interleaved", True):
            raise NotImplementedError("QwenAir reference supports only default interleaved MRoPE")
        kwargs["rope_theta"] = rope.get("rope_theta", text.get("rope_theta", cls.rope_theta))
        kwargs["partial_rotary_factor"] = rope.get(
            "partial_rotary_factor", text.get("partial_rotary_factor", cls.partial_rotary_factor)
        )
        kwargs["mrope_section"] = tuple(rope.get("mrope_section", cls.mrope_section))
        mtp = text.get("mtp") or {}
        kwargs["mtp_num_hidden_layers"] = text.get(
            "mtp_num_hidden_layers", mtp.get("num_hidden_layers", 0)
        )
        return cls(**kwargs)

    def validate(self) -> None:
        """Check the shape and scheduling invariants needed by QwenAir."""
        if self.num_hidden_layers < 1 or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")
        if set(self.layer_types) - {"linear_attention", "qwen_sparse_attention"}:
            raise ValueError("QwenAir supports only linear_attention and qwen_sparse_attention")
        positive = (
            self.vocab_size,
            self.hidden_size,
            self.head_dim,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_conv_kernel_dim,
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.hc_count,
            self.hc_lowrank,
            self.indexer_n_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
            self.max_single_rank_ple_elements,
            self.max_single_rank_parameters,
            self.max_reference_sequence_length,
            self.ple_conv_kernel_size,
            self.ngram_vocab_size_base,
            self.make_ngram_vocab_size_divisible_by,
            self.split_ngram_parts,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("QwenAir dimensions and resource limits must be positive")
        if self.hc_count <= 1 or self.num_experts_per_tok > self.num_experts:
            raise ValueError("Invalid hyper-connection or MoE top-k geometry")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must divide by num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must divide by linear_num_key_heads")
        if self.indexer_kv_heads != 1 or self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("QSA requires one index key head and an integral block budget")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim < 2 or rotary_dim % 2 or rotary_dim > self.indexer_head_dim:
            raise ValueError("RoPE dimension must be even and fit QSA index heads")
        if len(self.mrope_section) != 3 or sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError("mrope_section must partition the rotary dimension into three axes")
        if self.output_gate_type != "sigmoid":
            raise ValueError("Only the QwenAir sigmoid output gate is supported")
        if self.mamba_ssm_dtype != "float32":
            raise ValueError("QwenAir requires mamba_ssm_dtype=float32")
        if self.hidden_act != "silu":
            raise ValueError("Only the QwenAir SiLU activation is supported")
        if self.moe_expert_backend not in ("loop", "te_grouped"):
            raise ValueError("moe_expert_backend must be 'loop' or 'te_grouped'")
        if self.qsa_backend not in ("dense", "te_reference", "te_indexed_sdpa", "te_triton"):
            raise ValueError(
                "qsa_backend must be 'dense', 'te_reference', 'te_indexed_sdpa', or 'te_triton'"
            )
        if self.ple_layer_ids:
            heads = (self.ngram_size - 1) * self.heads_per_ngram
            if self.ngram_size < 2 or self.ple_embed_dim <= 0 or self.ple_embed_dim % heads:
                raise ValueError("PLE width must divide evenly across n-gram heads")
            if self.eos_token_id is None:
                raise ValueError("PLE requires eos_token_id")
            for layer in self.ple_layer_ids:
                if (
                    not 1 <= layer <= self.num_hidden_layers
                    or self.layer_types[layer - 1] != "linear_attention"
                ):
                    raise ValueError("PLE layer IDs must be one-based linear-attention layers")
        if self.mtp_num_hidden_layers not in (0, 1):
            raise ValueError("QwenAir MTP declaration permits at most one layer")
        parallel_sizes = (
            self.tensor_model_parallel_size,
            self.pipeline_model_parallel_size,
            self.context_parallel_size,
            self.expert_model_parallel_size,
            self.expert_tensor_parallel_size,
        )
        if any(size < 1 for size in parallel_sizes):
            raise ValueError("QwenAir parallel sizes must be positive")
        if any(
            size != 1
            for size in (
                self.tensor_model_parallel_size,
                self.pipeline_model_parallel_size,
                self.context_parallel_size,
            )
        ):
            raise NotImplementedError("QwenAir currently supports TP=PP=CP=1")
        if self.expert_tensor_parallel_size != 1:
            raise NotImplementedError("QwenAir currently supports expert TP size one")
        if self.num_experts % self.expert_model_parallel_size:
            raise ValueError("QwenAir expert count must divide evenly across EP")

    @property
    def rotary_dim(self) -> int:
        """Number of rotated coordinates in both full attention and QSA."""
        return int(self.head_dim * self.partial_rotary_factor)
