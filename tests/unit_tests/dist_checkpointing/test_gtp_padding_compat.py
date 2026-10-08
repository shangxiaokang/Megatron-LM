# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility tests for Bridge's GTP-aware checkpoint loading hooks."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing import serialization
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.utils import (
    grant_shape_mismatch_for_gtp_padding,
    resolve_gtp_pad_for_alignment,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 1),
        ({"fp8": True}, 16),
        ({"fp8_recipe": "mxfp8"}, 32),
        ({"fp4": True, "fp8_recipe": "mxfp8", "fp8": True}, 16),
    ],
)
def test_resolve_gtp_pad_for_alignment(kwargs, expected):
    assert resolve_gtp_pad_for_alignment(**kwargs) == expected


def _sharded_tensor(key: str, *, prepend_axis_num: int = 0) -> ShardedTensor:
    tensor = torch.empty(8, 2)
    offsets = ((0, 0, 1),) if prepend_axis_num else ()
    result = ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *offsets,
        prepend_axis_num=prepend_axis_num,
    )
    result.gtp_pad_length = 2
    return result


def test_grant_shape_mismatch_accepts_only_gtp_padding(monkeypatch):
    unpadded = _sharded_tensor("unpadded")
    aligned = _sharded_tensor("aligned")
    invalid = _sharded_tensor("invalid")
    already_allowed = _sharded_tensor("already_allowed")
    already_allowed.allow_shape_mismatch = True
    prepended = _sharded_tensor("prepended", prepend_axis_num=1)
    missing = _sharded_tensor("missing")
    metadata = {
        "unpadded": SimpleNamespace(global_shape=(6, 2)),
        "aligned": SimpleNamespace(global_shape=(16, 2)),
        "invalid": SimpleNamespace(global_shape=(7, 2)),
        "already_allowed": SimpleNamespace(global_shape=(7, 2)),
        "prepended": SimpleNamespace(global_shape=(1, 6, 2)),
    }
    monkeypatch.setattr(serialization, "load_tensors_metadata", lambda _path: metadata)

    grant_shape_mismatch_for_gtp_padding(
        {
            "first": [unpadded, aligned, invalid],
            "second": {"allowed": already_allowed, "prepended": prepended, "missing": missing},
        },
        "/unused/checkpoint",
        pad_for_alignment=16,
    )

    assert unpadded.allow_shape_mismatch
    assert aligned.allow_shape_mismatch
    assert prepended.allow_shape_mismatch
    assert already_allowed.allow_shape_mismatch
    assert not invalid.allow_shape_mismatch
    assert not missing.allow_shape_mismatch


def test_grant_shape_mismatch_tolerates_unreadable_metadata(monkeypatch, caplog):
    sharded_tensor = _sharded_tensor("weight")

    def fail_metadata(_path):
        raise OSError("unavailable")

    monkeypatch.setattr(serialization, "load_tensors_metadata", fail_metadata)
    grant_shape_mismatch_for_gtp_padding(
        {"weight": sharded_tensor}, "/missing/checkpoint", pad_for_alignment=16
    )

    assert not sharded_tensor.allow_shape_mismatch
    assert "could not read metadata" in caplog.text
