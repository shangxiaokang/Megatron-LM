# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional ragged Transformer Engine backend for QwenAir packed experts.

The reference loop owns the public parameter contract.  This adapter keeps the
same two parameters and attaches them to unregistered TE operation shells, so
enabling grouped execution does not introduce ``weight0`` ... ``weightN`` keys.
"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .layers import QwenAirExperts


_SINGLE_WEIGHT_ENV = "NVTE_GROUPED_LINEAR_SINGLE_PARAM"
_MIN_TE_VERSION = "2.14.0"


class QwenAirDispatchedExpertBackend:
    """Execute expert-sorted token rows with the loop or TE grouped GEMMs.

    The TE path is deliberately strict.  It is prepared only after the owning
    module has moved to CUDA BF16 and before DDP or an optimizer is constructed.
    Unsupported TE versions and devices fail instead of silently selecting
    discrete per-expert weights or the reference loop.
    """

    def __init__(self, experts: QwenAirExperts, backend: str) -> None:
        if backend not in ("loop", "te_grouped"):
            raise ValueError("QwenAir expert backend must be 'loop' or 'te_grouped'")
        self.experts = experts
        self.backend = backend
        self._ops: tuple[Any, ...] | None = None
        self._prepared_parameter_ids: tuple[int, int] | None = None
        self._te: Any = None
        self._grouped_tensor_type: type | None = None
        self._grouped_path_supported: Any = None
        if backend == "te_grouped":
            self._load_te_api()

    def _load_te_api(self) -> None:
        """Import and validate the experimental TE APIs used by this backend."""
        try:
            single_weight_enabled = int(os.environ.get(_SINGLE_WEIGHT_ENV, "0")) > 0
        except ValueError as exc:
            raise RuntimeError(f"{_SINGLE_WEIGHT_ENV} must be an integer") from exc
        if not single_weight_enabled:
            raise RuntimeError(
                "QwenAir te_grouped experts require "
                f"{_SINGLE_WEIGHT_ENV}=1; TE otherwise silently creates per-expert parameters"
            )

        try:
            import transformer_engine.pytorch as te
            from transformer_engine.pytorch.ops.basic.grouped_linear import (
                is_op_fuser_grouped_tensor_path_supported,
            )
            from transformer_engine.pytorch.tensor import GroupedTensor

            from megatron.core.utils import is_te_min_version
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "QwenAir te_grouped experts require a working Transformer Engine installation"
            ) from exc

        if not is_te_min_version(_MIN_TE_VERSION):
            raise RuntimeError(
                f"QwenAir te_grouped experts require Transformer Engine >= {_MIN_TE_VERSION}"
            )
        required_ops = ("GroupedLinear", "Sequential", "SwiGLU")
        missing = [name for name in required_ops if not hasattr(te.ops, name)]
        if missing or not hasattr(GroupedTensor, "make_grouped_tensor_from_rowwise_data"):
            detail = ", ".join(missing) if missing else "GroupedTensor rowwise wrapper"
            raise RuntimeError(f"Transformer Engine is missing required QwenAir APIs: {detail}")
        grouped_linear_args = inspect.signature(te.ops.GroupedLinear.__init__).parameters
        if "single_grouped_weight" not in grouped_linear_args:
            raise RuntimeError(
                "Transformer Engine GroupedLinear lacks single_grouped_weight support"
            )
        wrapper_args = inspect.signature(
            GroupedTensor.make_grouped_tensor_from_rowwise_data
        ).parameters
        expected_wrapper_args = {"num_tensors", "tensor_shape", "rowwise_data", "internal"}
        if not expected_wrapper_args.issubset(wrapper_args):
            raise RuntimeError(
                "Transformer Engine GroupedTensor rowwise wrapper has an incompatible signature"
            )

        self._te = te
        self._grouped_tensor_type = GroupedTensor
        self._grouped_path_supported = is_op_fuser_grouped_tensor_path_supported

    @property
    def is_prepared(self) -> bool:
        """Whether the TE operation shells reference the current expert parameters."""
        if self.backend == "loop":
            return True
        ids = (id(self.experts.gate_up_proj), id(self.experts.down_proj))
        return self._ops is not None and self._prepared_parameter_ids == ids

    def invalidate_for_module_apply(self) -> None:
        """Release TE shells before ``Module._apply`` moves their parameters."""
        if self.backend == "loop":
            return
        if self._ops is not None:
            for parameter in (self.experts.gate_up_proj, self.experts.down_proj):
                if parameter.grad is not None or getattr(parameter, "main_grad", None) is not None:
                    raise RuntimeError(
                        "QwenAir te_grouped parameters cannot be moved after DDP/optimizer setup"
                    )
        self._ops = None
        self._prepared_parameter_ids = None

    def prepare_after_module_apply(self) -> bool:
        """Wrap CUDA BF16 parameters and construct unregistered TE op shells.

        Returns ``False`` while the module is still on CPU or in a non-BF16
        dtype.  The explicit TE backend will reject execution until preparation
        has succeeded.
        """
        if self.backend == "loop":
            return True
        if self.is_prepared:
            return True

        gate_up = self.experts.gate_up_proj
        down = self.experts.down_proj
        if gate_up.device != down.device or gate_up.dtype != down.dtype:
            raise RuntimeError("QwenAir packed expert parameters must share device and dtype")
        if gate_up.device.type != "cuda" or gate_up.dtype != torch.bfloat16:
            return False

        with torch.cuda.device(gate_up.device):
            if not self._grouped_path_supported(None, torch.bfloat16):
                raise RuntimeError(
                    "Transformer Engine native BF16 grouped-tensor GEMM is unsupported on this GPU"
                )

        grouped_type = self._grouped_tensor_type
        if grouped_type is None:
            raise RuntimeError("Transformer Engine GroupedTensor API was not initialized")
        gate_is_grouped = isinstance(gate_up, grouped_type)
        down_is_grouped = isinstance(down, grouped_type)
        if gate_is_grouped != down_is_grouped:
            raise RuntimeError("QwenAir packed expert parameters have inconsistent TE storage")

        if gate_is_grouped:
            grouped_gate_up = gate_up
            grouped_down = down
        else:
            grouped_gate_up = self._wrap_parameter(gate_up, tuple(gate_up.shape[1:]))
            grouped_down = self._wrap_parameter(down, tuple(down.shape[1:]))

        ops = self._make_ops(grouped_gate_up, grouped_down)
        if not gate_is_grouped:
            self.experts.gate_up_proj = grouped_gate_up
            self.experts.down_proj = grouped_down
        self._ops = (ops,)
        self._prepared_parameter_ids = (id(self.experts.gate_up_proj), id(self.experts.down_proj))
        return True

    def _wrap_parameter(
        self, parameter: nn.Parameter, member_shape: tuple[int, int]
    ) -> nn.Parameter:
        """Wrap one contiguous packed parameter without changing its public shape."""
        if parameter.grad is not None or getattr(parameter, "main_grad", None) is not None:
            raise RuntimeError("QwenAir expert parameters must be wrapped before training setup")
        if not parameter.is_contiguous():
            raise RuntimeError("QwenAir packed expert parameters must be contiguous")
        num_experts = parameter.shape[0]
        grouped = self._grouped_tensor_type.make_grouped_tensor_from_rowwise_data(
            num_tensors=num_experts,
            tensor_shape=member_shape,
            rowwise_data=parameter.detach(),
            dtype=parameter.dtype,
            internal=False,
        )
        wrapped = nn.Parameter(grouped, requires_grad=parameter.requires_grad)
        for name, value in vars(parameter).items():
            if name not in vars(wrapped):
                setattr(wrapped, name, value)
        if tuple(wrapped.shape) != tuple(parameter.shape):
            raise RuntimeError("TE GroupedTensor changed the QwenAir packed parameter shape")
        if (
            wrapped.rowwise_data.untyped_storage().data_ptr()
            != parameter.untyped_storage().data_ptr()
        ):
            raise RuntimeError("TE GroupedTensor unexpectedly copied the QwenAir parameter")
        return wrapped

    def _make_grouped_linear(
        self, weight: nn.Parameter, in_features: int, out_features: int
    ) -> Any:
        """Attach an existing packed parameter to a meta-device TE op shell."""
        op = self._te.ops.GroupedLinear(
            weight.shape[0],
            in_features,
            out_features,
            bias=False,
            device="meta",
            dtype=weight.dtype,
            single_grouped_weight=True,
        )
        if not op.single_grouped_weight:
            raise RuntimeError("Transformer Engine disabled single_grouped_weight")
        op.register_parameter("weight", weight)
        for expert in range(op.num_groups):
            op.register_parameter(f"weight{expert}", None)
        return op

    def _make_ops(self, gate_up: nn.Parameter, down: nn.Parameter) -> Any:
        """Build FC1, unscaled SwiGLU, and FC2 without registering new model params."""
        hidden_size = gate_up.shape[2]
        intermediate_size = down.shape[2]
        if gate_up.shape[1] != 2 * intermediate_size or down.shape[1] != hidden_size:
            raise RuntimeError("QwenAir packed expert parameter geometry is inconsistent")
        fc1 = self._make_grouped_linear(gate_up, hidden_size, 2 * intermediate_size)
        fc2 = self._make_grouped_linear(down, intermediate_size, hidden_size)
        return self._te.ops.Sequential(fc1, self._te.ops.SwiGLU(), fc2)

    def __call__(self, hidden: Tensor, tokens_per_expert: Tensor, scores: Tensor) -> Tensor:
        """Apply the selected backend to contiguous expert-sorted rows."""
        if self.backend == "loop":
            return self.experts.forward_dispatched(hidden, tokens_per_expert, scores)
        if not self.is_prepared:
            raise RuntimeError(
                "QwenAir te_grouped experts must be moved to CUDA BF16 before DDP/optimizer setup"
            )
        if hidden.device.type != "cuda" or hidden.dtype != torch.bfloat16:
            raise RuntimeError("QwenAir te_grouped experts accept only CUDA BF16 activations")
        if tokens_per_expert.device.type != "cpu" or tokens_per_expert.dtype != torch.int64:
            raise RuntimeError("QwenAir te_grouped experts require CPU int64 dispatcher counts")
        expert_count = self.experts.gate_up_proj.shape[0]
        if tokens_per_expert.ndim != 1 or tokens_per_expert.numel() != expert_count:
            raise ValueError("Dispatched QwenAir expert counts have the wrong shape")
        counts = tokens_per_expert.tolist()
        if any(count < 0 for count in counts) or sum(counts) != hidden.shape[0]:
            raise ValueError("Dispatched QwenAir expert counts do not match input rows")
        if scores.numel() != hidden.shape[0] or scores.device != hidden.device:
            raise ValueError("Dispatched QwenAir expert scores have the wrong shape or device")
        if self.experts.gate_up_proj.device != hidden.device:
            raise RuntimeError("QwenAir expert weights and activations must share a CUDA device")

        device_counts = tokens_per_expert.to(device=hidden.device)
        (ops,) = self._ops
        ops.train(self.experts.training)
        output = ops(hidden.contiguous(), device_counts, device_counts)
        if tuple(output.shape) != tuple(hidden.shape):
            raise RuntimeError("Transformer Engine returned an invalid QwenAir expert output shape")
        weighted = output * scores.reshape(-1).unsqueeze(-1)
        return weighted.to(hidden.dtype)
