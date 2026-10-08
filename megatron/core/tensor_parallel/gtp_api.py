# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fail-closed compatibility surface for generalized tensor parallelism.

This MCore baseline predates the GTP runtime.  Newer Megatron-Bridge releases
import ``HAVE_GTP`` even for ordinary DDP checkpoint loading, so expose the
capability flag without pretending that GTP operators are available.
"""

HAVE_GTP = False


__all__ = ["HAVE_GTP"]
