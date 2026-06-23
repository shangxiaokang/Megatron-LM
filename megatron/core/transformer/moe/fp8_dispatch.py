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


def _all_to_all_single_async(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
) -> tuple[torch.Tensor, object]:
    """Asynchronous all-to-all single. Caller owns stream/event synchronization."""
    input_ = input_.contiguous()
    output = torch.empty(
        [sum(output_split_sizes), *input_.shape[1:]],
        dtype=input_.dtype,
        device=input_.device,
    )
    work = torch.distributed.all_to_all_single(
        output,
        input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    return output, work


def _prefix_offsets(sizes: Sequence[int]) -> list[int]:
    """Exclusive prefix offsets for split-size lists."""
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += int(size)
    return offsets


def _stage_split_sizes(
    sizes: Sequence[int], stage: int, num_stages: int
) -> tuple[list[int], list[int]]:
    """Return per-peer row starts and row counts for a pipeline stage."""
    starts = []
    stage_sizes = []
    for size in sizes:
        size = int(size)
        base = size // num_stages
        remainder = size % num_stages
        stage_size = base + (1 if stage < remainder else 0)
        start = stage * base + min(stage, remainder)
        starts.append(start)
        stage_sizes.append(stage_size)
    return starts, stage_sizes


def _pack_split_rows(
    tensor: torch.Tensor,
    split_sizes: Sequence[int],
    stage_starts: Sequence[int],
    stage_sizes: Sequence[int],
) -> torch.Tensor:
    """Pack the selected rows from every peer chunk into an all-to-all input buffer."""
    total_rows = sum(stage_sizes)
    packed = torch.empty(
        [total_rows, *tensor.shape[1:]],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    src_offsets = _prefix_offsets(split_sizes)
    dst_offset = 0
    for src_offset, stage_start, stage_size in zip(src_offsets, stage_starts, stage_sizes):
        if stage_size == 0:
            continue
        src_start = src_offset + stage_start
        src_end = src_start + stage_size
        packed[dst_offset : dst_offset + stage_size].copy_(tensor[src_start:src_end])
        dst_offset += stage_size
    return packed


def _copy_split_rows_(
    output: torch.Tensor,
    packed: torch.Tensor,
    split_sizes: Sequence[int],
    stage_starts: Sequence[int],
    stage_sizes: Sequence[int],
) -> None:
    """Scatter a packed stage output back into full all-to-all output order."""
    dst_offsets = _prefix_offsets(split_sizes)
    src_offset = 0
    for dst_offset, stage_start, stage_size in zip(dst_offsets, stage_starts, stage_sizes):
        if stage_size == 0:
            continue
        dst_start = dst_offset + stage_start
        dst_end = dst_start + stage_size
        output[dst_start:dst_end].copy_(packed[src_offset : src_offset + stage_size])
        src_offset += stage_size


def _empty_blockwise_payload(num_rows: int, hidden_size: int, device: torch.device):
    """Create empty rowwise blockwise FP8 payload buffers for zero-row sends."""
    return (
        torch.empty((num_rows, hidden_size), dtype=torch.uint8, device=device),
        torch.empty((num_rows, (hidden_size + 127) // 128), dtype=torch.float32, device=device),
    )

def _recipe_fp8_dtype(fprop_tensor: bool):
    """Return the TE FP8 dtype that matches the active FP8 recipe."""
    if not HAVE_TE_BLOCKWISE_FP8:
        raise ImportError(
            "Transformer Engine blockwise FP8 support is required for MoE FP8 token "
            "communication."
        )
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    return get_fp8_te_dtype(recipe, fprop_tensor=fprop_tensor)


def _validate_blockwise_fp8_input(input_: torch.Tensor, fp8_dtype, op_name: str) -> None:
    """Validate pre-quantized compact rowwise blockwise FP8 payloads."""
    if input_._rowwise_data is None or input_._rowwise_scale_inv is None:
        raise RuntimeError(f"{op_name} got a blockwise FP8 tensor without rowwise data/scales.")
    if input_._is_2D_scaled:
        raise RuntimeError(f"{op_name} only supports 1D blockwise FP8 tensors.")
    if input_._data_format != tex.Float8BlockScaleTensorFormat.COMPACT:
        raise RuntimeError(f"{op_name} requires COMPACT blockwise FP8 tensors.")
    if input_._fp8_dtype != fp8_dtype:
        raise RuntimeError(f"{op_name} got FP8 dtype {input_._fp8_dtype}, expected {fp8_dtype}.")


def _all_to_all_blockwise_fp8_pipelined(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]],
    input_split_sizes: Optional[Sequence[int]],
    fp8_dtype,
    quantizer,
    op_name: str,
    dequantize_output: bool,
) -> Optional[torch.Tensor]:
    """Two-stage pipelined blockwise FP8 all-to-all.

    This first version splits every peer chunk along rows, so each rank receives two
    partial outputs in the same final peer order. It can overlap stage-1 quantization
    with stage-0 communication, and stage-0 dequant/scatter with stage-1 communication.
    """
    num_stages = 2
    world_size = group.size()
    output_split_sizes = _normalize_split_sizes(output_split_sizes)
    input_split_sizes = _normalize_split_sizes(input_split_sizes)
    if (
        world_size < num_stages
        or output_split_sizes is None
        or input_split_sizes is None
        or len(output_split_sizes) != world_size
        or len(input_split_sizes) != world_size
    ):
        return None

    hidden_size = input_.shape[1]
    if sum(input_split_sizes) != input_.shape[0]:
        return None

    input_is_fp8 = isinstance(input_, Float8BlockwiseQTensor)
    if input_is_fp8:
        _validate_blockwise_fp8_input(input_, fp8_dtype, op_name)
        input_dtype = input_.dtype
        input_data = input_._rowwise_data
        input_scale_inv = input_._rowwise_scale_inv
        requires_grad = input_.requires_grad
    else:
        input_dtype = input_.dtype
        input_data = None
        input_scale_inv = None
        requires_grad = input_.requires_grad

    device = input_.device
    current_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device=device)
    post_stream = torch.cuda.Stream(device=device)

    if dequantize_output:
        final_output = torch.empty(
            (sum(output_split_sizes), hidden_size), dtype=input_dtype, device=device
        )
        final_output.record_stream(post_stream)
        final_data = None
        final_scale_inv = None
    else:
        final_data, final_scale_inv = _empty_blockwise_payload(
            sum(output_split_sizes), hidden_size, device
        )
        final_data.record_stream(post_stream)
        final_scale_inv.record_stream(post_stream)
        final_output = None

    stages = []
    for stage in range(num_stages):
        input_starts, input_stage_sizes = _stage_split_sizes(
            input_split_sizes, stage, num_stages
        )
        output_starts, output_stage_sizes = _stage_split_sizes(
            output_split_sizes, stage, num_stages
        )

        with torch.cuda.stream(current_stream):
            if input_is_fp8:
                send_data = _pack_split_rows(
                    input_data, input_split_sizes, input_starts, input_stage_sizes
                )
                send_scale_inv = _pack_split_rows(
                    input_scale_inv, input_split_sizes, input_starts, input_stage_sizes
                )
            else:
                input_part = _pack_split_rows(
                    input_, input_split_sizes, input_starts, input_stage_sizes
                )
                if input_part.size(0) == 0:
                    send_data, send_scale_inv = _empty_blockwise_payload(0, hidden_size, device)
                else:
                    input_part_fp8 = quantizer(input_part.contiguous())
                    send_data = input_part_fp8._rowwise_data
                    send_scale_inv = input_part_fp8._rowwise_scale_inv
            quant_event = torch.cuda.Event()
            quant_event.record(current_stream)

        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(quant_event)
            recv_data, data_work = _all_to_all_single_async(
                group, send_data, output_stage_sizes, input_stage_sizes
            )
            recv_scale_inv, scale_work = _all_to_all_single_async(
                group, send_scale_inv, output_stage_sizes, input_stage_sizes
            )
            comm_event = torch.cuda.Event()
            comm_event.record(comm_stream)

        with torch.cuda.stream(post_stream):
            post_stream.wait_event(comm_event)
            recv_data.record_stream(post_stream)
            recv_scale_inv.record_stream(post_stream)
            if dequantize_output:
                if sum(output_stage_sizes) == 0:
                    output_part = torch.empty((0, hidden_size), dtype=input_dtype, device=device)
                else:
                    recv_fp8 = Float8BlockwiseQTensor(
                        shape=recv_data.shape,
                        dtype=input_dtype,
                        rowwise_data=recv_data,
                        rowwise_scale_inv=recv_scale_inv,
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        fp8_dtype=fp8_dtype,
                        quantizer=quantizer,
                        is_2D_scaled=False,
                        data_format=tex.Float8BlockScaleTensorFormat.COMPACT,
                        requires_grad=requires_grad,
                    )
                    output_part = recv_fp8.dequantize(dtype=input_dtype)
                _copy_split_rows_(
                    final_output,
                    output_part,
                    output_split_sizes,
                    output_starts,
                    output_stage_sizes,
                )
            else:
                _copy_split_rows_(
                    final_data,
                    recv_data,
                    output_split_sizes,
                    output_starts,
                    output_stage_sizes,
                )
                _copy_split_rows_(
                    final_scale_inv,
                    recv_scale_inv,
                    output_split_sizes,
                    output_starts,
                    output_stage_sizes,
                )
            post_event = torch.cuda.Event()
            post_event.record(post_stream)

        stages.append(
            {
                "works": (data_work, scale_work),
                "send_data": send_data,
                "send_scale_inv": send_scale_inv,
                "recv_data": recv_data,
                "recv_scale_inv": recv_scale_inv,
                "post_event": post_event,
            }
        )

    for stage in stages:
        for work in stage["works"]:
            work.wait()
        current_stream.wait_event(stage["post_event"])

    if dequantize_output:
        return final_output
    return Float8BlockwiseQTensor(
        shape=final_data.shape,
        dtype=input_dtype,
        rowwise_data=final_data,
        rowwise_scale_inv=final_scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=fp8_dtype,
        quantizer=quantizer,
        is_2D_scaled=False,
        data_format=tex.Float8BlockScaleTensorFormat.COMPACT,
        requires_grad=requires_grad,
    )


def _all_to_all_blockwise_fp8(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]],
    input_split_sizes: Optional[Sequence[int]],
    fp8_dtype,
    op_name: str,
    dequantize_output: bool = True,
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
    pipelined_output = _all_to_all_blockwise_fp8_pipelined(
        group,
        input_,
        output_split_sizes,
        input_split_sizes,
        fp8_dtype=fp8_dtype,
        quantizer=quantizer,
        op_name=op_name,
        dequantize_output=dequantize_output,
    )
    if pipelined_output is not None:
        return pipelined_output

    if isinstance(input_, Float8BlockwiseQTensor):
        _validate_blockwise_fp8_input(input_, fp8_dtype, op_name)
        input_fp8 = input_
        input_dtype = input_.dtype
    else:
        input_fp8 = quantizer(input_.contiguous())
        input_dtype = input_.dtype

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
        dtype=input_dtype,
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
    if dequantize_output:
        return recv_fp8.dequantize(dtype=input_dtype)
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
        dequantize_output: bool = True,
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
            dequantize_output=dequantize_output,
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


def get_fp8_dispatch_dtype():
    """Return the active recipe's forward FP8 dtype for MoE dispatch payloads."""
    return _recipe_fp8_dtype(fprop_tensor=True)


def get_fp8_combine_backward_dtype():
    """Return the active recipe's backward FP8 dtype for MoE combine gradients."""
    return _recipe_fp8_dtype(fprop_tensor=False)


def all_to_all_blockwise_fp8_dispatch(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
    dequantize_output: bool = True,
):
    """All-to-all dispatch using recipe-aligned blockwise FP8 forward payload.

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
        dequantize_output,
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
