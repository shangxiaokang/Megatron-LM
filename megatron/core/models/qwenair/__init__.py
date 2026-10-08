# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir text reference model for staged Megatron-Core integration."""

from .config import QwenAirTextConfig
from .model import QwenAirForCausalLM, QwenAirModel, QwenAirOutput, QwenAirTextModel
from .qsa import QwenAirQSAIndexer, QwenAirQSASelection, qsa_dense_attention

__all__ = [
    "QwenAirTextConfig",
    "QwenAirTextModel",
    "QwenAirForCausalLM",
    "QwenAirModel",
    "QwenAirOutput",
    "QwenAirQSAIndexer",
    "QwenAirQSASelection",
    "qsa_dense_attention",
]
