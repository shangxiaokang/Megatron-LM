# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FP8 communication helpers for MoE token dispatch."""

from typing import Optional, Sequence

import torch

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )

    HAVE_TE_BLOCKWISE_FP8 = True
except ImportError:
    tex = None
    Float8BlockQuantizer = None
    Float8BlockwiseQTensor = None
    HAVE_TE_BLOCKWISE_FP8 = False


def _normalize_split_sizes(split_sizes: Optional[Sequence[int]]) -> Optional[list[int]]:
    """Convert split sizes to a plain Python list for torch.distributed."""
    if split_sizes is None:
        return None
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.detach().cpu().tolist()
    elif hasattr(split_sizes, "tolist"):
        split_sizes = split_sizes.tolist()
    return [int(size) for size in split_sizes]


def _all_to_all_single(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]],
    input_split_sizes: Optional[Sequence[int]],
) -> torch.Tensor:
    """All-to-all single without attaching its own autograd rule."""
    world_size = group.size()
    if world_size == 1:
        return input_

    output_split_sizes = _normalize_split_sizes(output_split_sizes)
    input_split_sizes = _normalize_split_sizes(input_split_sizes)

    input_ = input_.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(input_)
    else:
        output = torch.empty(
            [sum(output_split_sizes), *input_.shape[1:]],
            dtype=input_.dtype,
            device=input_.device,
        )

    torch.distributed.all_to_all_single(
        output,
        input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


class _AllToAllBlockwiseFP8Dispatch(torch.autograd.Function):
    """Forward all-to-all with blockwise FP8 payload and high-precision backward."""

    @staticmethod
    def forward(
        ctx,
        group: torch.distributed.ProcessGroup,
        input_: torch.Tensor,
        output_split_sizes: Optional[Sequence[int]],
        input_split_sizes: Optional[Sequence[int]],
        fp8_dtype,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = _normalize_split_sizes(output_split_sizes)
        ctx.input_split_sizes = _normalize_split_sizes(input_split_sizes)

        if group.size() == 1:
            return input_

        if not HAVE_TE_BLOCKWISE_FP8:
            raise ImportError(
                "Transformer Engine blockwise FP8 support is required for "
                "moe_token_dispatcher_fp8."
            )
        if not input_.is_cuda:
            raise RuntimeError("FP8 token dispatch requires a CUDA tensor.")
        if input_.dim() != 2:
            raise RuntimeError(
                f"FP8 token dispatch expects a 2D [tokens, hidden] tensor, got {input_.shape}."
            )

        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=False,
            amax_epsilon=0.0,
            force_pow_2_scales=True,
            block_scaling_dim=1,
            all_gather_usage=True,
        )
        input_fp8 = quantizer(input_.contiguous())

        recv_data = _all_to_all_single(
            group,
            input_fp8._rowwise_data,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
        )
        recv_scale_inv = _all_to_all_single(
            group,
            input_fp8._rowwise_scale_inv,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
        )

        recv_fp8 = Float8BlockwiseQTensor(
            shape=recv_data.shape,
            dtype=input_.dtype,
            rowwise_data=recv_data,
            rowwise_scale_inv=recv_scale_inv,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=fp8_dtype,
            quantizer=quantizer,
            is_2D_scaled=False,
            data_format=tex.Float8BlockScaleTensorFormat.COMPACT,
        )
        return recv_fp8.dequantize(dtype=input_.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_single(
            ctx.group,
            grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
        )
        return None, grad_input, None, None, None


def all_to_all_blockwise_fp8_dispatch(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
):
    """All-to-all dispatch using FP8 E4M3 blockwise quantized forward payload.

    The backward path intentionally mirrors the original high-precision all-to-all
    so this helper only changes forward dispatch communication.
    """
    if not HAVE_TE_BLOCKWISE_FP8:
        raise ImportError(
            "Transformer Engine blockwise FP8 support is required for "
            "moe_token_dispatcher_fp8."
        )
    return _AllToAllBlockwiseFP8Dispatch.apply(
        group,
        input_,
        output_split_sizes,
        input_split_sizes,
        tex.DType.kFloat8E4M3,
    )
