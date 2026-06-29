# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FP8 communication helpers for MoE token exchange."""

from typing import Optional, Sequence

import torch

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, get_fp8_te_dtype
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )

    HAVE_TE_BLOCKWISE_FP8 = True
except ImportError:
    tex = None
    FP8GlobalStateManager = None
    get_fp8_te_dtype = None
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


def is_blockwise_fp8_qtensor(input_: torch.Tensor) -> bool:
    """Return whether the tensor is a TE blockwise FP8 QTensor."""
    return HAVE_TE_BLOCKWISE_FP8 and isinstance(input_, Float8BlockwiseQTensor)


def _make_blockwise_fp8_qtensor_like(
    input_: torch.Tensor,
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
) -> torch.Tensor:
    """Construct a blockwise FP8 QTensor after token-dimension-only movement."""
    return Float8BlockwiseQTensor(
        shape=rowwise_data.shape,
        dtype=input_.dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=input_._fp8_dtype,
        quantizer=input_._quantizer,
        is_2D_scaled=input_._is_2D_scaled,
        data_format=input_._data_format,
        requires_grad=input_.requires_grad,
    )


def _validate_blockwise_fp8_rowwise_compact(input_: torch.Tensor, op_name: str) -> None:
    """Validate the compact rowwise blockwise FP8 layout used by token exchange."""
    if not is_blockwise_fp8_qtensor(input_):
        raise TypeError(f"{op_name} expects a Float8BlockwiseQTensor.")
    if input_._rowwise_data is None or input_._rowwise_scale_inv is None:
        raise RuntimeError(f"{op_name} requires rowwise FP8 data and scales.")
    if input_._columnwise_data is not None or input_._columnwise_scale_inv is not None:
        raise RuntimeError(f"{op_name} does not support columnwise FP8 data.")
    if input_.dim() != 2:
        raise RuntimeError(f"{op_name} expects a 2D [tokens, hidden] tensor, got {input_.shape}.")
    if input_._data_format != tex.Float8BlockScaleTensorFormat.COMPACT:
        raise RuntimeError(f"{op_name} expects COMPACT blockwise FP8 scale format.")
    if input_._is_2D_scaled:
        raise RuntimeError(f"{op_name} expects 1D block scaling.")
    if input_._rowwise_data.shape[0] != input_._rowwise_scale_inv.shape[0]:
        raise RuntimeError(f"{op_name} expects data and scale rows to match token rows.")


def sort_blockwise_fp8_qtensor_by_idxs(
    input_: torch.Tensor,
    split_sizes: Sequence[int],
    sorted_idxs: Sequence[int],
) -> torch.Tensor:
    """Sort a blockwise FP8 QTensor by token chunks.

    The compact rowwise blockwise scale tensor is laid out as [tokens, hidden_blocks], so token
    chunk movement must be applied to both FP8 data and scale rows.
    """
    _validate_blockwise_fp8_rowwise_compact(input_, "Blockwise FP8 chunk sort")
    split_sizes = _normalize_split_sizes(split_sizes)
    sorted_idxs = _normalize_split_sizes(sorted_idxs)

    data_chunks = torch.split(input_._rowwise_data, split_sizes, dim=0)
    scale_chunks = torch.split(input_._rowwise_scale_inv, split_sizes, dim=0)
    rowwise_data = torch.cat([data_chunks[i] for i in sorted_idxs], dim=0)
    rowwise_scale_inv = torch.cat([scale_chunks[i] for i in sorted_idxs], dim=0)
    return _make_blockwise_fp8_qtensor_like(input_, rowwise_data, rowwise_scale_inv)


def transpose_blockwise_fp8_qtensor_token_groups(
    input_: torch.Tensor,
    first_dim: int,
    second_dim: int,
    third_dim: int,
) -> torch.Tensor:
    """Apply view(first, second, third, hidden).transpose(0, 1).flatten(0, 2).

    This mirrors the alltoall dispatcher drop-and-pad token reorder while keeping rowwise FP8
    scales aligned with token rows.
    """
    _validate_blockwise_fp8_rowwise_compact(input_, "Blockwise FP8 token group transpose")
    hidden = input_._rowwise_data.shape[1:]
    scale_hidden = input_._rowwise_scale_inv.shape[1:]

    rowwise_data = (
        input_._rowwise_data.view(first_dim, second_dim, third_dim, *hidden)
        .transpose(0, 1)
        .contiguous()
        .flatten(start_dim=0, end_dim=2)
    )
    rowwise_scale_inv = (
        input_._rowwise_scale_inv.view(first_dim, second_dim, third_dim, *scale_hidden)
        .transpose(0, 1)
        .contiguous()
        .flatten(start_dim=0, end_dim=2)
    )
    return _make_blockwise_fp8_qtensor_like(input_, rowwise_data, rowwise_scale_inv)


def _recipe_fp8_dtype(fprop_tensor: bool):
    """Return the TE FP8 dtype that matches the active FP8 recipe."""
    if not HAVE_TE_BLOCKWISE_FP8:
        raise ImportError(
            "Transformer Engine blockwise FP8 support is required for MoE FP8 token "
            "communication."
        )
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    return get_fp8_te_dtype(recipe, fprop_tensor=fprop_tensor)


def _all_to_all_blockwise_fp8(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]],
    input_split_sizes: Optional[Sequence[int]],
    fp8_dtype,
    op_name: str,
    dequantize: bool = True,
) -> torch.Tensor:
    """All-to-all with blockwise FP8 payload and optional BF16/FP16 output."""
    if group.size() == 1:
        return input_

    if not HAVE_TE_BLOCKWISE_FP8:
        raise ImportError(
            "Transformer Engine blockwise FP8 support is required for MoE FP8 token "
            "communication."
        )
    if not input_.is_cuda:
        raise RuntimeError(f"{op_name} requires a CUDA tensor.")
    if input_.dim() != 2:
        raise RuntimeError(f"{op_name} expects a 2D [tokens, hidden] tensor, got {input_.shape}.")

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

    output_split_sizes = _normalize_split_sizes(output_split_sizes)
    input_split_sizes = _normalize_split_sizes(input_split_sizes)

    recv_data = _all_to_all_single(
        group,
        input_fp8._rowwise_data,
        output_split_sizes,
        input_split_sizes,
    )
    recv_scale_inv = _all_to_all_single(
        group,
        input_fp8._rowwise_scale_inv,
        output_split_sizes,
        input_split_sizes,
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
        requires_grad=input_.requires_grad,
    )
    if dequantize:
        return recv_fp8.dequantize(dtype=input_.dtype)
    return recv_fp8


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
        dequantize: bool,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = _normalize_split_sizes(output_split_sizes)
        ctx.input_split_sizes = _normalize_split_sizes(input_split_sizes)

        return _all_to_all_blockwise_fp8(
            group,
            input_,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            fp8_dtype=fp8_dtype,
            op_name="FP8 token dispatch",
            dequantize=dequantize,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_single(
            ctx.group,
            grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
        )
        return None, grad_input, None, None, None, None


class _AllToAllBlockwiseFP8CombineBackward(torch.autograd.Function):
    """High-precision forward all-to-all with blockwise FP8 backward payload."""

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
        ctx.fp8_dtype = fp8_dtype
        return _all_to_all_single(group, input_, ctx.output_split_sizes, ctx.input_split_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_blockwise_fp8(
            ctx.group,
            grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            fp8_dtype=ctx.fp8_dtype,
            op_name="FP8 token combine backward",
        )
        return None, grad_input, None, None, None


def all_to_all_blockwise_fp8_dispatch(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
    dequantize: bool = True,
):
    """All-to-all dispatch using recipe-aligned blockwise FP8 forward payload.

    If dequantize is false, the receive side returns the reconstructed blockwise FP8 QTensor.
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
        _recipe_fp8_dtype(fprop_tensor=True),
        dequantize,
    )


def all_to_all_blockwise_fp8_combine_backward(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
):
    """All-to-all combine with original-precision forward and FP8 backward communication.

    The backward FP8 dtype follows the active TE recipe: E5M2 for hybrid recipes and E4M3
    for E4M3 recipes.
    """
    if not HAVE_TE_BLOCKWISE_FP8:
        raise ImportError(
            "Transformer Engine blockwise FP8 support is required for "
            "moe_token_combine_backward_fp8."
        )
    return _AllToAllBlockwiseFP8CombineBackward.apply(
        group,
        input_,
        output_split_sizes,
        input_split_sizes,
        _recipe_fp8_dtype(fprop_tensor=False),
    )
