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


def _normalize_split_matrix(
    split_matrix: Sequence[Sequence[int]], world_size: int, num_local_experts: int
) -> list[list[int]]:
    """Convert [world_size, num_local_experts] split sizes to a Python matrix."""
    if isinstance(split_matrix, torch.Tensor):
        split_matrix = split_matrix.detach().cpu().tolist()
    elif hasattr(split_matrix, "tolist"):
        split_matrix = split_matrix.tolist()

    if len(split_matrix) != world_size:
        raise RuntimeError(
            f"Expected split matrix with {world_size} rows, got {len(split_matrix)}."
        )

    normalized_matrix = []
    for row in split_matrix:
        if len(row) != num_local_experts:
            raise RuntimeError(
                f"Expected split matrix rows with {num_local_experts} entries, got {len(row)}."
            )
        normalized_matrix.append([int(size) for size in row])
    return normalized_matrix


def _make_blockwise_quantizer(fp8_dtype):
    return Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=1,
        all_gather_usage=True,
    )


def _dequantize_blockwise_fp8(
    recv_data: torch.Tensor,
    recv_scale_inv: torch.Tensor,
    quantizer,
    fp8_dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    recv_fp8 = Float8BlockwiseQTensor(
        shape=recv_data.shape,
        dtype=output_dtype,
        rowwise_data=recv_data,
        rowwise_scale_inv=recv_scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=fp8_dtype,
        quantizer=quantizer,
        is_2D_scaled=False,
        data_format=tex.Float8BlockScaleTensorFormat.COMPACT,
    )
    return recv_fp8.dequantize(dtype=output_dtype)


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
    output_split_sizes: Optional[Sequence[int]],
    input_split_sizes: Optional[Sequence[int]],
) -> tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Async all-to-all single without attaching its own autograd rule."""
    world_size = group.size()
    if world_size == 1:
        return input_, None

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

    work = torch.distributed.all_to_all_single(
        output,
        input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    return output, work


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
) -> torch.Tensor:
    """All-to-all with blockwise FP8 payload and BF16/FP16 output."""
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

    quantizer = _make_blockwise_quantizer(fp8_dtype)
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

    return _dequantize_blockwise_fp8(
        recv_data,
        recv_scale_inv,
        quantizer=quantizer,
        fp8_dtype=fp8_dtype,
        output_dtype=input_.dtype,
    )


def _partition_by_expert_split(
    input_: torch.Tensor,
    split_matrix: list[list[int]],
    expert_split_index: int,
    first_partition: bool,
) -> tuple[torch.Tensor, list[int]]:
    """Pack either half of each rank chunk according to local expert boundaries."""
    pieces = []
    partition_split_sizes = []
    offset = 0
    for row in split_matrix:
        rank_total = sum(row)
        first_count = sum(row[:expert_split_index])
        partition_count = first_count if first_partition else rank_total - first_count
        partition_offset = 0 if first_partition else first_count
        rank_chunk = input_.narrow(0, offset, rank_total)
        pieces.append(rank_chunk.narrow(0, partition_offset, partition_count))
        partition_split_sizes.append(partition_count)
        offset += rank_total

    if offset != input_.size(0):
        raise RuntimeError(
            f"Expert split matrix describes {offset} tokens, but input has {input_.size(0)}."
        )
    return torch.cat(pieces, dim=0), partition_split_sizes


def _stitch_expert_partitions(
    first_partition: torch.Tensor,
    second_partition: torch.Tensor,
    first_split_sizes: Sequence[int],
    second_split_sizes: Sequence[int],
) -> torch.Tensor:
    """Restore [rank0 first+second, rank1 first+second, ...] chunk order."""
    pieces = []
    first_offset = 0
    second_offset = 0
    for first_count, second_count in zip(first_split_sizes, second_split_sizes):
        pieces.append(first_partition.narrow(0, first_offset, first_count))
        pieces.append(second_partition.narrow(0, second_offset, second_count))
        first_offset += first_count
        second_offset += second_count
    return torch.cat(pieces, dim=0)


def _wait_all(works: Sequence[Optional[torch.distributed.Work]]) -> None:
    for work in works:
        if work is not None:
            work.wait()


def _launch_fp8_partition_all_to_all(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    fp8_dtype,
):
    quantizer = _make_blockwise_quantizer(fp8_dtype)
    input_fp8 = quantizer(input_.contiguous())
    recv_data, data_work = _all_to_all_single_async(
        group,
        input_fp8._rowwise_data,
        output_split_sizes,
        input_split_sizes,
    )
    recv_scale_inv, scale_work = _all_to_all_single_async(
        group,
        input_fp8._rowwise_scale_inv,
        output_split_sizes,
        input_split_sizes,
    )
    return {
        "input": input_,
        "input_fp8": input_fp8,
        "quantizer": quantizer,
        "recv_data": recv_data,
        "recv_scale_inv": recv_scale_inv,
        "works": (data_work, scale_work),
    }


def _all_to_all_blockwise_fp8_by_expert_split(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes_by_expert: Sequence[Sequence[int]],
    input_split_sizes_by_expert: Sequence[Sequence[int]],
    num_local_experts: int,
    fp8_dtype,
    op_name: str,
) -> torch.Tensor:
    """All-to-all with two expert partitions to overlap FP8 quant/dequant with comm."""
    world_size = group.size()
    if world_size == 1:
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
    if num_local_experts < 2:
        raise RuntimeError(f"{op_name} expert split requires at least two local experts.")

    expert_split_index = num_local_experts // 2
    input_split_matrix = _normalize_split_matrix(
        input_split_sizes_by_expert, world_size, num_local_experts
    )
    output_split_matrix = _normalize_split_matrix(
        output_split_sizes_by_expert, world_size, num_local_experts
    )

    input_ = input_.contiguous()
    first_input, first_input_splits = _partition_by_expert_split(
        input_,
        input_split_matrix,
        expert_split_index,
        first_partition=True,
    )
    first_output_splits = [sum(row[:expert_split_index]) for row in output_split_matrix]
    first_pending = _launch_fp8_partition_all_to_all(
        group,
        first_input,
        first_output_splits,
        first_input_splits,
        fp8_dtype,
    )

    second_input, second_input_splits = _partition_by_expert_split(
        input_,
        input_split_matrix,
        expert_split_index,
        first_partition=False,
    )
    second_output_splits = [sum(row[expert_split_index:]) for row in output_split_matrix]
    second_pending = _launch_fp8_partition_all_to_all(
        group,
        second_input,
        second_output_splits,
        second_input_splits,
        fp8_dtype,
    )

    _wait_all(first_pending["works"])
    first_output = _dequantize_blockwise_fp8(
        first_pending["recv_data"],
        first_pending["recv_scale_inv"],
        quantizer=first_pending["quantizer"],
        fp8_dtype=fp8_dtype,
        output_dtype=input_.dtype,
    )

    _wait_all(second_pending["works"])
    second_output = _dequantize_blockwise_fp8(
        second_pending["recv_data"],
        second_pending["recv_scale_inv"],
        quantizer=second_pending["quantizer"],
        fp8_dtype=fp8_dtype,
        output_dtype=input_.dtype,
    )

    return _stitch_expert_partitions(
        first_output,
        second_output,
        first_output_splits,
        second_output_splits,
    )


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

        return _all_to_all_blockwise_fp8(
            group,
            input_,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            fp8_dtype=fp8_dtype,
            op_name="FP8 token dispatch",
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_single(
            ctx.group,
            grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
        )
        return None, grad_input, None, None, None


class _AllToAllBlockwiseFP8DispatchByExpertSplit(torch.autograd.Function):
    """Forward FP8 all-to-all split by local experts; high-precision backward."""

    @staticmethod
    def forward(
        ctx,
        group: torch.distributed.ProcessGroup,
        input_: torch.Tensor,
        output_split_sizes: Optional[Sequence[int]],
        input_split_sizes: Optional[Sequence[int]],
        output_split_sizes_by_expert: Sequence[Sequence[int]],
        input_split_sizes_by_expert: Sequence[Sequence[int]],
        num_local_experts: int,
        fp8_dtype,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = _normalize_split_sizes(output_split_sizes)
        ctx.input_split_sizes = _normalize_split_sizes(input_split_sizes)

        return _all_to_all_blockwise_fp8_by_expert_split(
            group,
            input_,
            output_split_sizes_by_expert,
            input_split_sizes_by_expert,
            num_local_experts,
            fp8_dtype=fp8_dtype,
            op_name="FP8 token dispatch expert split",
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_single(
            ctx.group,
            grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
        )
        return None, grad_input, None, None, None, None, None, None


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
    )


def all_to_all_blockwise_fp8_dispatch_by_expert_split(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: Optional[Sequence[int]] = None,
    input_split_sizes: Optional[Sequence[int]] = None,
    output_split_sizes_by_expert: Optional[Sequence[Sequence[int]]] = None,
    input_split_sizes_by_expert: Optional[Sequence[Sequence[int]]] = None,
    num_local_experts: int = 0,
):
    """FP8 dispatch with payload split into two local-expert partitions."""
    if not HAVE_TE_BLOCKWISE_FP8:
        raise ImportError(
            "Transformer Engine blockwise FP8 support is required for "
            "moe_token_dispatcher_fp8."
        )
    if output_split_sizes_by_expert is None or input_split_sizes_by_expert is None:
        raise RuntimeError("Expert split FP8 dispatch requires per-expert split metadata.")
    return _AllToAllBlockwiseFP8DispatchByExpertSplit.apply(
        group,
        input_,
        output_split_sizes,
        input_split_sizes,
        output_split_sizes_by_expert,
        input_split_sizes_by_expert,
        num_local_experts,
        _recipe_fp8_dtype(fprop_tensor=True),
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
