# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QwenAir text reference model for staged Megatron-Core integration."""

from .config import QwenAirTextConfig
from .distributed import (
    QwenAirMemoryEstimate,
    QwenAirParallelTopology,
    QwenAirProcessGroups,
    build_qwenair_process_groups,
    estimate_qwenair_training_memory,
    plan_qwenair_parallel_topology,
)
from .model import QwenAirForCausalLM, QwenAirModel, QwenAirOutput, QwenAirTextModel
from .qsa import QwenAirQSAIndexer, QwenAirQSASelection, qsa_dense_attention

__all__ = [
    "QwenAirTextConfig",
    "QwenAirParallelTopology",
    "QwenAirProcessGroups",
    "QwenAirMemoryEstimate",
    "plan_qwenair_parallel_topology",
    "build_qwenair_process_groups",
    "estimate_qwenair_training_memory",
    "QwenAirTextModel",
    "QwenAirForCausalLM",
    "QwenAirModel",
    "QwenAirOutput",
    "QwenAirQSAIndexer",
    "QwenAirQSASelection",
    "qsa_dense_attention",
]
