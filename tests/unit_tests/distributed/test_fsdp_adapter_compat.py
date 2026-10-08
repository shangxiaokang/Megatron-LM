# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-Bridge compatibility names for the MCore FSDP adapter."""

import pytest

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel,
    FullyShardedDataParallelV1,
    FullyShardedDataParallelV2,
)


def test_original_fsdp_adapter_is_the_v1_compatibility_type():
    assert FullyShardedDataParallelV1 is FullyShardedDataParallel


def test_unavailable_fsdp_v2_fails_closed():
    with pytest.raises(NotImplementedError, match="FSDP v2 is unavailable"):
        FullyShardedDataParallelV2()
