# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Topology-independent initialization for QwenAir parameter shards."""

from __future__ import annotations

import torch
from torch import Tensor, nn

_MASK64 = (1 << 64) - 1
_MASK63 = (1 << 63) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_DEFAULT_CHUNK_ELEMENTS = 1 << 20


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _namespace_hash(namespace: str) -> int:
    value = _FNV_OFFSET
    for byte in namespace.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK64
    return value


def qwenair_shard_seed(
    base_seed: int, namespace: str, layer_idx: int, global_index: int
) -> int:
    """Derive a stable torch seed from a logical parameter identity."""
    value = _splitmix64((base_seed & _MASK64) ^ _namespace_hash(namespace))
    value = _splitmix64(value ^ (layer_idx & _MASK64))
    value = _splitmix64(value ^ (global_index & _MASK64))
    return value & _MASK63


def _local_generator(tensor: Tensor, seed: int) -> torch.Generator:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    return generator


def initialize_qwenair_expert_normal_(
    tensor: Tensor,
    *,
    base_seed: int,
    namespace: str,
    layer_idx: int,
    first_global_expert: int,
    std: float,
) -> Tensor:
    """Initialize packed experts by logical expert ID without global RNG use."""
    if tensor.is_meta:
        return tensor
    if tensor.ndim < 1:
        raise ValueError("QwenAir expert tensors need a leading expert axis")
    for local_expert, expert_tensor in enumerate(tensor.unbind(0)):
        seed = qwenair_shard_seed(
            base_seed, namespace, layer_idx, first_global_expert + local_expert
        )
        nn.init.normal_(expert_tensor, std=std, generator=_local_generator(tensor, seed))
    return tensor


def initialize_qwenair_sharded_normal_(
    tensor: Tensor,
    *,
    global_element_start: int,
    logical_numel: int,
    base_seed: int,
    namespace: str,
    layer_idx: int,
    std: float,
    chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS,
) -> Tensor:
    """Initialize a flat logical tensor identically under any contiguous sharding.

    Fixed-size logical chunks, rather than runtime rank shards, own independent
    RNG streams. Boundary ranks materialize at most one chunk temporarily, so
    concatenating shards is bitwise equal to initializing the unsharded tensor.
    """
    if tensor.is_meta:
        return tensor
    if global_element_start < 0 or logical_numel < 0 or chunk_elements < 1:
        raise ValueError("QwenAir shard initialization received invalid bounds")
    local_numel = tensor.numel()
    global_element_end = global_element_start + local_numel
    if global_element_end > logical_numel:
        raise ValueError("QwenAir shard extends beyond its logical tensor")
    if local_numel == 0:
        return tensor

    flat = tensor.view(-1)
    first_chunk = global_element_start // chunk_elements
    last_chunk = (global_element_end - 1) // chunk_elements
    with torch.no_grad():
        for chunk_index in range(first_chunk, last_chunk + 1):
            chunk_start = chunk_index * chunk_elements
            chunk_end = min(chunk_start + chunk_elements, logical_numel)
            overlap_start = max(global_element_start, chunk_start)
            overlap_end = min(global_element_end, chunk_end)
            destination_start = overlap_start - global_element_start
            destination_end = overlap_end - global_element_start
            seed = qwenair_shard_seed(base_seed, namespace, layer_idx, chunk_index)
            generator = _local_generator(tensor, seed)
            if overlap_start == chunk_start and overlap_end == chunk_end:
                nn.init.normal_(
                    flat[destination_start:destination_end], std=std, generator=generator
                )
                continue
            chunk = torch.empty(
                chunk_end - chunk_start, device=tensor.device, dtype=tensor.dtype
            )
            nn.init.normal_(chunk, std=std, generator=generator)
            source_start = overlap_start - chunk_start
            source_end = overlap_end - chunk_start
            flat[destination_start:destination_end].copy_(chunk[source_start:source_end])
    return tensor
