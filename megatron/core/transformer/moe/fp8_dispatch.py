# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""FP8 AlltoAll communication utilities for MoE token dispatch/combine.

This module provides FP8 quantized AlltoAll communication to reduce MoE
inter-rank communication volume by 2x compared to BF16.

Key features:
- TE C++ Float8BlockQuantizer (cached singleton) for 1D blockwise activation
  quantization (block_scaling_dim=1, scale [N, H//128])
- Triton fused dequantize kernel: one program per row, grid=(N,), looping over
  H//128 blocks with static_range (matches TE's `_dequantize_vectorwise` math)
- Fallback to Triton quantize+dequantize when TE unavailable
- Hybrid format support: E4M3 forward, E5M2 backward (straight-through estimator
  for backward re-quantizes gradients)

Communication flow:
1. 1D blockwise quantize: hidden_states [N, H] BF16 -> FP8 uint8 [N, H] + scales
   [N, H//128] FP32 via TE C++
2. AlltoAll FP8 data as uint8 (halved communication volume)
3. AlltoAll scales [N, H//128] (per-token splits, same shape)
4. Dequantize: Triton row kernel -> BF16
"""

import torch
import torch.distributed as dist
from typing import Union, Tuple, Optional

# TE blockwise FP8 support
_HAVE_TE_BLOCKWISE = False
try:
    import transformer_engine
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )
    _HAVE_TE_BLOCKWISE = True
except ImportError:
    tex = None
    Float8BlockQuantizer = None
    Float8BlockwiseQTensor = None

# Triton: fused dequantize kernels (and fallback quantize when TE unavailable)
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# FP8 dtype constants
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FP8_E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0

# Default block size for blockwise quantization (same as TE)
DEFAULT_BLOCK_SIZE = 128

# Cached Float8BlockQuantizer singletons (1D blockwise activation quantization,
# block_scaling_dim=1 -> scale [N, H//128]). Constructed lazily so importing this
# module never triggers TE init; reused across calls instead of rebuilding per
# dispatch/combine (forward + backward = 2 calls per MoE layer).
_TE_QUANTIZER_CACHE = {}


def _get_te_quantizer(use_e5m2: bool):
    """Return a cached Float8BlockQuantizer for the requested FP8 format."""
    key = use_e5m2
    quantizer = _TE_QUANTIZER_CACHE.get(key)
    if quantizer is None:
        fp8_dtype = tex.DType.kFloat8E5M2 if use_e5m2 else tex.DType.kFloat8E4M3
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=False,
            amax_epsilon=0.0,
            force_pow_2_scales=True,
            block_scaling_dim=1,
            all_gather_usage=True,
        )
        _TE_QUANTIZER_CACHE[key] = quantizer
    return quantizer


# ---------------------------------------------------------------------------
# Triton fused dequantize kernel (one program per row, 1D blockwise scale)
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    # ---------------------------------------------------------------------------
    # Triton fused dequantize: one program per row (1D blockwise, scale [N, H//128])
    # ---------------------------------------------------------------------------

    @triton.jit
    def _triton_dequantize_row_kernel(
        data_ptr,      # [N, H] fp8 (raw bits viewed as fp8 dtype in Python)
        scale_ptr,     # [N, H//128] FP32 (1D blockwise scale, one per 128 cols)
        output_ptr,    # [N, H] bf16
        N,
        H: tl.constexpr,            # hidden size, fixed per model run (e.g. 2048)
        NUM_BLOCKS: tl.constexpr,   # H // 128 (e.g. 16)
        BLOCK_SIZE: tl.constexpr,   # 128, matches TE block size
        USE_E5M2: tl.constexpr,
    ):
        """Fused blockwise dequantize: one program per row, whole-row load.

        Each program loads the ENTIRE row of H fp8 values in a single large
        transaction (1 load of H bytes) instead of NUM_BLOCKS separate 128B loads.
        Larger transactions raise memory-level parallelism and bandwidth
        utilization -- this is the dominant lever on high-BW devices (H800), where
        the per-block-load variant only reached ~40% of peak bandwidth.

        Scale is loaded as a vector of NUM_BLOCKS values indexed by col//BLOCK_SIZE,
        broadcast across each 128-element block. Matches TE `_dequantize_vectorwise`
        math bit-for-bit:
            result[n, k] = fp8[n, k].to(f32) * scale_inv[n, k//128]  -> bf16
        """
        pid_row = tl.program_id(0)  # which row (0..N-1)
        row_off = pid_row * H

        col = tl.arange(0, H)                          # [H], whole row
        if USE_E5M2:
            fp8_val = tl.load(data_ptr + row_off + col).to(tl.float8e5)
        else:
            fp8_val = tl.load(data_ptr + row_off + col).to(tl.float8e4nv)
        x_fp32 = fp8_val.to(tl.float32)

        # Vectorized scale load: scale_inv[n, col//128]. Loading all NUM_BLOCKS
        # scales (with repetition) in one op beats NUM_BLOCKS scalar loads.
        scale = tl.load(scale_ptr + pid_row * NUM_BLOCKS + col // BLOCK_SIZE).to(tl.float32)
        x_dequant = x_fp32 * scale

        tl.store(output_ptr + row_off + col, x_dequant.to(tl.bfloat16))

    # ---------------------------------------------------------------------------
    # Triton whole-row quantize: one program per row (1D blockwise, scale [N, H//128])
    # ---------------------------------------------------------------------------

    @triton.jit
    def _triton_quantize_row_kernel(
        input_ptr,      # [N, H] bf16/fp32
        fp8_ptr,        # [N, H] uint8 (fp8 bits)
        scale_inv_ptr,  # [N, H//128] f32  (stores 1/scale, same as TE)
        N,
        H: tl.constexpr,            # hidden size, fixed per model run (e.g. 2048)
        NUM_BLOCKS: tl.constexpr,   # H // 128
        BLOCK_SIZE: tl.constexpr,   # 128
        FP8_MAX: tl.constexpr,      # 448.0 (E4M3) / 57344.0 (E5M2)
        USE_E5M2: tl.constexpr,
    ):
        """Fused blockwise quantize: one program per row, whole-row load.

        Mirrors the math of TE's `block_scaled_1d_cast_transpose_kernel` rowwise path
        but skips the shared-memory staging (which TE needs only for the optional
        transpose). Computing amax+scale+cast straight from registers reaches ~88%
        bandwidth (like the dequant kernel) vs TE's ~54%.

        Scale math (matches TE `compute_scale_from_amax`):
            amax = max(|x|) per 128-block
            scale = fp8_max / amax      (amax==0 -> scale=1)
            force_pow_2_scales: scale &= 0xFF800000  (clear mantissa -> pow2)
            scale_inv = 1 / scale        (stored, consumed by dequant as x*scale_inv)
        Quantized value stored as:  x * scale  (== x / amax * fp8_max)
        """
        pid = tl.program_id(0)
        ro = pid * H
        col = tl.arange(0, H)                          # whole row
        x = tl.load(input_ptr + ro + col).to(tl.float32)

        # per-block amax via reshape (NUM_BLOCKS, BLOCK_SIZE) then reduce axis=1
        x_blk = tl.reshape(x, (NUM_BLOCKS, BLOCK_SIZE))
        amax = tl.max(tl.abs(x_blk), axis=1)          # [NUM_BLOCKS]

        # scale = fp8_max / amax; amax==0 (or inf/nan) -> scale=1 (TE behavior)
        scale = FP8_MAX / amax
        scale = tl.where(amax == 0.0, 1.0, scale)

        # force_pow_2_scales=True: clear mantissa bits -> round down to power of 2.
        # float32 -> int32 (bitcast) -> & 0xFF800000 -> float32 (bitcast).
        # 0xFF800000 == 4286578688 exceeds int32 range, so use the two's-complement
        # int32 value -8388608 (same bit pattern). Matches TE's `scale_bits &= 0xFF800000`.
        scale_bits = scale.to(tl.int32, bitcast=True) & (-8388608)
        scale = scale_bits.to(tl.float32, bitcast=True)
        scale_inv = 1.0 / scale

        # broadcast scale[NUM_BLOCKS] -> [H]: reshape (NUM_BLOCKS,1) -> broadcast
        # to (NUM_BLOCKS, BLOCK_SIZE) -> flatten (H,). Each 128-block uses its scale.
        scale_h = tl.reshape(scale, (NUM_BLOCKS, 1))
        scale_h = tl.broadcast_to(scale_h, (NUM_BLOCKS, BLOCK_SIZE))
        scale_h = tl.reshape(scale_h, (H,))
        xs = x * scale_h
        if USE_E5M2:
            out = xs.to(tl.float8e5)
        else:
            out = xs.to(tl.float8e4nv)
        # Store fp8 bits as uint8 (output buffer is uint8). bitcast fp8 -> uint8 avoids
        # the fp8->uint8 implicit cast that Triton 3.x rejects on store.
        tl.store(fp8_ptr + ro + col, out.to(tl.uint8, bitcast=True))

        # store scale_inv [NUM_BLOCKS]
        tl.store(scale_inv_ptr + pid * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS), scale_inv)


# ---------------------------------------------------------------------------
# TE-based quantize
# ---------------------------------------------------------------------------

def _quantize_te(tensor: torch.Tensor, block_size: int, use_e5m2: bool):
    """TE C++ blockwise FP8 quantization using a cached Float8BlockQuantizer.

    1D blockwise activation quantization (block_scaling_dim=1):
        fp8_data: uint8 [N, H]
        scale_inv: FP32 [N, H//128]  (one scale per 128 cols per row)
    """
    quantizer = _get_te_quantizer(use_e5m2)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Explicit quantize() avoids the __call__ dispatch indirection. Under
    # autograd.Function.forward/backward grad is disabled, so this takes the
    # _QuantizeFunc.forward path (no extra autograd wrapper).
    input_fp8 = quantizer.quantize(tensor)

    # Extract raw components for A2A
    fp8_data = input_fp8._rowwise_data        # uint8 [N, H]
    scale_inv = input_fp8._rowwise_scale_inv  # FP32 [N, H//128] 1D blockwise

    return fp8_data, scale_inv


# ---------------------------------------------------------------------------
# Triton dequantize for TE 2D scale_inv format
# ---------------------------------------------------------------------------

def _dequantize_te(fp8_data: torch.Tensor, scale_inv: torch.Tensor,
                   block_size: int, use_e5m2: bool):
    """Triton fused dequantize for TE 1D blockwise scale_inv format.

    Args:
        fp8_data: fp8 [N, H] (already viewed as float8 dtype)
        scale_inv: FP32 [N, H//128] 1D blockwise scale (one scale per 128 cols per row)
        block_size: 128 (matches TE block size)
        use_e5m2: True for E5M2 format

    Returns:
        Dequantized tensor [N, H] in BF16
    """
    N, H = fp8_data.shape
    num_blocks = H // block_size

    output = torch.empty(N, H, dtype=torch.bfloat16, device=fp8_data.device)

    grid = (N,)
    _triton_dequantize_row_kernel[grid](
        fp8_data, scale_inv, output,
        N,
        H=H,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
        USE_E5M2=use_e5m2,
        num_warps=2,
        num_stages=2,
    )
    return output


# ---------------------------------------------------------------------------
# PyTorch fallback quantize (when TE unavailable)
# ---------------------------------------------------------------------------

def _quantize_py(tensor: torch.Tensor, block_size: int, use_e5m2: bool):
    """PyTorch blockwise FP8 quantization fallback."""
    N, H = tensor.shape
    assert H % block_size == 0
    num_blocks = H // block_size
    fp8_max = FP8_E5M2_MAX if use_e5m2 else FP8_E4M3_MAX
    fp8_dtype = torch.float8_e5m2 if use_e5m2 else torch.float8_e4m3fn

    tensor_blocked = tensor.view(N, num_blocks, block_size)
    amax = tensor_blocked.abs().amax(dim=2)
    scale = amax / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    tensor_scaled = tensor_blocked / scale.unsqueeze(2)
    tensor_clamped = tensor_scaled.clamp(-fp8_max, fp8_max)
    fp8_blocked = tensor_clamped.to(fp8_dtype)

    return fp8_blocked.view(N, H), scale


def _dequantize_py(fp8_data: torch.Tensor, scales: torch.Tensor,
                    block_size: int, use_e5m2: bool):
    """PyTorch blockwise FP8 dequantization fallback."""
    N, H = fp8_data.shape
    num_blocks = H // block_size
    fp8_dtype = torch.float8_e5m2 if use_e5m2 else torch.float8_e4m3fn

    fp8_blocked = fp8_data.view(fp8_dtype).view(N, num_blocks, block_size).float()
    tensor_dequant = fp8_blocked * scales.unsqueeze(2)
    return tensor_dequant.view(N, H).bfloat16()


# ---------------------------------------------------------------------------
# Triton whole-row quantize (preferred; TE quant kept but not used by default)
# ---------------------------------------------------------------------------

def _quantize_triton_row(tensor: torch.Tensor, block_size: int, use_e5m2: bool):
    """Triton whole-row blockwise FP8 quantization.

    One program per row, whole-row load, amax+scale+cast fused in registers.
    Matches TE output format: fp8_data uint8 [N, H], scale_inv f32 [N, H//128].

    Faster than TE's `block_scaled_1d_cast_transpose_kernel` for the dispatch/combine
    path because TE's unified cast+transpose kernel unconditionally stages data in
    shared memory (for the transpose it doesn't need here, columnwise=False), capping
    bandwidth at ~54%; this rowwise-only kernel reaches ~88% (same as the dequant kernel).
    """
    N, H = tensor.shape
    num_blocks = H // block_size
    fp8_max = FP8_E5M2_MAX if use_e5m2 else FP8_E4M3_MAX

    # Output as uint8 (matches TE output; downstream A2A sends bytes, dequant re-views
    # to fp8). The kernel casts fp8 bits back to uint8 before storing (see kernel) so the
    # store target is a plain uint8 pointer -- robust across Triton versions.
    fp8_data = torch.empty(N, H, dtype=torch.uint8, device=tensor.device)
    scale_inv = torch.empty(N, num_blocks, dtype=torch.float32, device=tensor.device)

    grid = (N,)
    _triton_quantize_row_kernel[grid](
        tensor, fp8_data, scale_inv, N,
        H=H, NUM_BLOCKS=num_blocks, BLOCK_SIZE=block_size,
        FP8_MAX=fp8_max, USE_E5M2=use_e5m2,
        num_warps=2, num_stages=2,
    )
    return fp8_data, scale_inv


# ---------------------------------------------------------------------------
# Unified quantize/dequantize API
# ---------------------------------------------------------------------------

def _quantize_blockwise(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_e5m2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blockwise FP8 quantization along hidden dimension (1D blockwise, scale [N, H//128]).

    Priority: Triton whole-row (~bandwidth-bound) > TE (kept, not used by default) > PyTorch.
    Returns:
        fp8_data: uint8 [N, H]
        scales: FP32 [N, H//128]  (1D blockwise scale_inv, one per 128 cols per row)
    """
    assert tensor.is_contiguous(), "Input tensor must be contiguous"
    N, H = tensor.shape
    assert H % block_size == 0, f"Hidden size {H} must be divisible by block_size {block_size}"

    if _HAS_TRITON:
        return _quantize_triton_row(tensor, block_size, use_e5m2)
    if _HAVE_TE_BLOCKWISE:
        # TE's block_scaled_1d_cast_transpose_kernel unconditionally stages input to
        # shared memory (for the transpose path) even when columnwise=False, so the
        # rowwise-only dispatch/combine quant only reaches ~54% bandwidth. Kept as a
        # fallback for environments without Triton, but the Triton row kernel is
        # preferred (reaches ~88%, matching the dequant kernel).
        return _quantize_te(tensor, block_size, use_e5m2)
    return _quantize_py(tensor, block_size, use_e5m2)


def _dequantize_blockwise(
    fp8_data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_e5m2: bool = False,
) -> torch.Tensor:
    """Blockwise FP8 dequantization along hidden dimension.

    Scale is always 1D blockwise [N, H//128] f32 (one per 128 cols per row),
    produced by either the Triton row kernel or TE. One Triton program per row
    -> far fewer launches than a per-block grid.
    """
    N, H = fp8_data.shape
    assert H % block_size == 0

    # Reinterpret uint8 tensor as float8 dtype for Triton kernel.
    # TE quantize returns uint8, but Triton tl.load on a uint8 pointer yields uint8
    # values that cannot be directly cast to fp8 types. View as float8 dtype so the
    # pointer type is correct for the kernel.
    if fp8_data.dtype == torch.uint8:
        fp8_dtype = torch.float8_e5m2 if use_e5m2 else torch.float8_e4m3fn
        fp8_data = fp8_data.contiguous().view(fp8_dtype)

    if _HAS_TRITON:
        return _dequantize_te(fp8_data, scales, block_size, use_e5m2)
    return _dequantize_py(fp8_data, scales, block_size, use_e5m2)


# ---------------------------------------------------------------------------
# AlltoAll with FP8 quantized communication
# ---------------------------------------------------------------------------

def _alltoall_fp8(
    group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
    output_splits: Union[torch.Tensor, list],
    input_splits: Union[torch.Tensor, list],
    use_e5m2: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Perform AlltoAll with FP8 blockwise quantized communication.

    Steps:
    1. Blockwise quantize input to FP8 (TE C++ preferred)
    2. AlltoAll the FP8 data (as uint8, halving communication volume)
    3. AlltoAll the scales (per-token blockwise scales)
    4. Dequantize received data back to input dtype (Triton fused kernel)

    Args:
        group: EP process group.
        input_tensor: [N, H] input tensor (BF16/FP32).
        output_splits: [EP] tokens to receive from each rank.
        input_splits: [EP] tokens to send to each rank.
        use_e5m2: Use E5M2 format instead of E4M3.
        block_size: Block size for quantization (default 128).

    Returns:
        [M, H] tensor where M = sum(output_splits), same dtype as input.
    """
    world_size = group.size()

    if world_size == 1:
        return input_tensor

    input_dtype = input_tensor.dtype
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    N, H = input_tensor.shape

    # Step 1: Blockwise quantize to FP8
    fp8_data, local_scales = _quantize_blockwise(input_tensor, block_size, use_e5m2)

    # Convert splits to Python int lists
    if isinstance(output_splits, torch.Tensor):
        out_splits_list = [int(x) for x in output_splits.cpu().tolist()]
    else:
        out_splits_list = [int(x) for x in output_splits]

    if isinstance(input_splits, torch.Tensor):
        in_splits_list = [int(x) for x in input_splits.cpu().tolist()]
    else:
        in_splits_list = [int(x) for x in input_splits]

    sum_out = sum(out_splits_list)
    sum_in = sum(in_splits_list)

    # Megatron convention:
    #   input_splits[r]  = tokens this rank SENDS to rank r
    #   output_splits[r] = tokens this rank RECEIVES from rank r
    # PyTorch all_to_all_single(output, input, output_split_sizes, input_split_sizes):
    #   output_split_sizes describes output tensor (receive)
    #   input_split_sizes describes input tensor (send)
    assert N == sum_in, (
        f"FP8 AlltoAll: input_tensor.shape[0]={N} != sum(input_splits)={sum_in}, "
        f"in_splits={in_splits_list}"
    )

    # AlltoAll 1: FP8 data [N, H] -> [sum_out, H]
    recv_fp8 = torch.empty(sum_out, H, dtype=fp8_data.dtype, device=input_tensor.device)
    dist.all_to_all_single(
        recv_fp8, fp8_data.contiguous(),
        output_split_sizes=out_splits_list,
        input_split_sizes=in_splits_list,
        group=group,
    )

    # AlltoAll 2: scales -> split sizes depend on scale format
    # TE 2D format: [N//128, H//128] -> divide splits by block_size
    # Triton/py format: [N, H//128] -> use splits as-is
    if local_scales.shape[0] == N:
        # Non-TE format: per-token scales
        recv_scales = torch.empty(
            sum_out, local_scales.shape[1],
            dtype=local_scales.dtype, device=input_tensor.device,
        )
        scale_out_splits = out_splits_list
        scale_in_splits = in_splits_list
    else:
        # TE 2D tile format: per-block scales
        recv_scales = torch.empty(
            sum_out // block_size, local_scales.shape[1],
            dtype=local_scales.dtype, device=input_tensor.device,
        )
        scale_out_splits = [s // block_size for s in out_splits_list]
        scale_in_splits = [s // block_size for s in in_splits_list]

    dist.all_to_all_single(
        recv_scales, local_scales.contiguous(),
        output_split_sizes=scale_out_splits,
        input_split_sizes=scale_in_splits,
        group=group,
    )

    # Step 4: Dequantize back to original dtype
    output = _dequantize_blockwise(recv_fp8, recv_scales, block_size, use_e5m2)

    if input_dtype != torch.bfloat16:
        output = output.to(input_dtype)

    return output


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class _FP8AlltoAllFunction(torch.autograd.Function):
    """Autograd function for FP8 AlltoAll with hybrid format support.

    Forward: quantize via TE C++ -> A2A FP8 + scales -> Triton dequantize
    Backward: re-quantize gradient -> A2A -> dequantize (straight-through estimator)

    Only input_tensor is passed to apply() to avoid autograd engine position-mapping
    issues with non-tensor args (ProcessGroup, list, bool, int) under TE checkpoint
    re-run. Other params are stored in thread-local storage and picked up by forward.
    """

    import threading
    _tls = threading.local()

    @staticmethod
    def forward(ctx, input_tensor):
        tls = _FP8AlltoAllFunction._tls
        ctx.group = tls.group
        ctx.output_splits = tls.output_splits
        ctx.input_splits = tls.input_splits
        ctx.use_hybrid = tls.use_hybrid
        ctx.block_size = tls.block_size

        return _alltoall_fp8(
            ctx.group, input_tensor, ctx.output_splits, ctx.input_splits,
            use_e5m2=False, block_size=ctx.block_size
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _alltoall_fp8(
            ctx.group,
            grad_output.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            use_e5m2=ctx.use_hybrid,
            block_size=ctx.block_size,
        )

        return grad_input


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fp8_all_to_all(
    group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
    output_split_sizes: Union[torch.Tensor, list, None] = None,
    input_split_sizes: Union[torch.Tensor, list, None] = None,
    use_hybrid: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """FP8 AlltoAll wrapper matching Megatron's all_to_all interface.

    Provides 2x communication reduction by blockwise quantizing to FP8 before AlltoAll.
    Uses TE C++ Float8BlockQuantizer for quantize, Triton fused kernel for dequantize.

    Args:
        group: Process group for AlltoAll.
        input_tensor: [N, H] input tensor (BF16/FP32).
        output_split_sizes: Tokens to receive from each rank (default: equal splits).
        input_split_sizes: Tokens to send to each rank (default: equal splits).
        use_hybrid: If True, use E4M3 forward + E5M2 backward.
        block_size: Block size for blockwise quantization (default 128).

    Returns:
        [M, H] output tensor where M = sum(output_split_sizes), same dtype as input.
    """
    assert group is not None, "group should not be None"

    if output_split_sizes is None:
        world_size = group.size()
        total = input_tensor.shape[0]
        assert total % world_size == 0
        chunk = total // world_size
        output_split_sizes = [chunk] * world_size
        input_split_sizes = [chunk] * world_size

    # Normalize splits to Python lists before passing to autograd function.
    # This ensures autograd engine never sees tensor splits, avoiding "gradient
    # different than None at position N" errors under TE checkpoint re-run.
    if isinstance(output_split_sizes, torch.Tensor):
        output_split_sizes = output_split_sizes.detach().cpu().tolist()
    elif hasattr(output_split_sizes, 'tolist'):
        output_split_sizes = output_split_sizes.tolist()
    if isinstance(input_split_sizes, torch.Tensor):
        input_split_sizes = input_split_sizes.detach().cpu().tolist()
    elif hasattr(input_split_sizes, 'tolist'):
        input_split_sizes = input_split_sizes.tolist()

    # Store non-tensor params in thread-local storage, then call apply() with
    # only the tensor arg. This avoids autograd engine position-mapping issues
    # with non-tensor arguments (ProcessGroup, list, bool, int) and is safe
    # for potential future concurrent usage.
    tls = _FP8AlltoAllFunction._tls
    tls.group = group
    tls.output_splits = output_split_sizes
    tls.input_splits = input_split_sizes
    tls.use_hybrid = use_hybrid
    tls.block_size = block_size

    return _FP8AlltoAllFunction.apply(input_tensor)