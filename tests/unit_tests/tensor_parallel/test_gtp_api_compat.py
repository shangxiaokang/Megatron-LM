# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility contract for Bridge on a pre-GTP MCore baseline."""

from megatron.core.tensor_parallel import gtp_api


def test_pre_gtp_baseline_reports_the_feature_unavailable():
    assert gtp_api.HAVE_GTP is False
    assert gtp_api.__all__ == ["HAVE_GTP"]
