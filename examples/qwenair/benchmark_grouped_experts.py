# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark QwenAir's canonical EP32 local expert shard on one GPU.

This is a manual B300 gate, not a training recipe. It compares the existing
Python expert loop with the opt-in Transformer Engine ragged grouped backend
using 16 local experts and 1,280 dispatched rows (80 rows per expert).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch

from megatron.core.models.qwenair import QwenAirTextConfig
from megatron.core.models.qwenair.layers import QwenAirExperts
from megatron.core.models.qwenair.moe_grouped import QwenAirDispatchedExpertBackend


def _measure_ms(fn: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _forward_step(
    backend: QwenAirDispatchedExpertBackend,
    hidden: torch.Tensor,
    counts: torch.Tensor,
    scores: torch.Tensor,
) -> Callable[[], None]:
    def step() -> None:
        with torch.no_grad():
            backend(hidden, counts, scores)

    return step


def _training_step(
    experts: QwenAirExperts,
    backend: QwenAirDispatchedExpertBackend,
    hidden: torch.Tensor,
    counts: torch.Tensor,
    scores: torch.Tensor,
    grad_output: torch.Tensor,
) -> Callable[[], None]:
    def step() -> None:
        experts.zero_grad(set_to_none=True)
        hidden.grad = None
        backend(hidden, counts, scores).backward(grad_output)

    return step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The grouped expert benchmark requires CUDA")
    if os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM") != "1":
        raise RuntimeError("Set NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 for this explicit B300 gate")

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    config = QwenAirTextConfig()
    num_local_experts = config.num_experts // 32
    counts = torch.full((num_local_experts,), 80, dtype=torch.int64)
    num_tokens = int(counts.sum())
    hidden_data = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
    scores = torch.rand(num_tokens, device=device, dtype=torch.float32)
    grad_output = torch.randn_like(hidden_data)

    reference = QwenAirExperts(config, num_local_experts=num_local_experts)
    grouped = QwenAirExperts(config, num_local_experts=num_local_experts)
    grouped.load_state_dict(reference.state_dict(), strict=True)
    reference = reference.to(device=device, dtype=torch.bfloat16)
    grouped = grouped.to(device=device, dtype=torch.bfloat16)
    loop = QwenAirDispatchedExpertBackend(reference, "loop")
    grouped_backend = QwenAirDispatchedExpertBackend(grouped, "te_grouped")
    if not grouped_backend.prepare_after_module_apply():
        raise RuntimeError("Failed to prepare the CUDA BF16 TE grouped backend")

    with torch.no_grad():
        expected = loop(hidden_data, counts, scores)
        actual = grouped_backend(hidden_data, counts, scores)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=5e-3)

    reference_hidden = hidden_data.detach().clone().requires_grad_()
    parity_grouped_hidden = hidden_data.detach().clone().requires_grad_()
    expected = loop(reference_hidden, counts, scores)
    actual = grouped_backend(parity_grouped_hidden, counts, scores)
    expected.backward(grad_output)
    actual.backward(grad_output)
    torch.testing.assert_close(
        parity_grouped_hidden.grad.float(),
        reference_hidden.grad.float(),
        rtol=1e-2,
        atol=5e-3,
    )
    for name in ("gate_up_proj", "down_proj"):
        expected_grad = getattr(reference, name).grad
        actual_grad = getattr(grouped, name).grad
        if expected_grad is None or actual_grad is None:
            raise RuntimeError(f"Missing {name} gradient during grouped expert parity")
        torch.testing.assert_close(
            actual_grad.float(), expected_grad.float(), rtol=1e-2, atol=5e-3, msg=name
        )
    reference.zero_grad(set_to_none=True)
    grouped.zero_grad(set_to_none=True)

    loop_hidden = hidden_data.detach().clone().requires_grad_()
    grouped_hidden = hidden_data.detach().clone().requires_grad_()
    loop_forward_ms = _measure_ms(
        _forward_step(loop, loop_hidden, counts, scores), args.warmup, args.iterations
    )
    grouped_forward_ms = _measure_ms(
        _forward_step(grouped_backend, grouped_hidden, counts, scores), args.warmup, args.iterations
    )
    loop_training_ms = _measure_ms(
        _training_step(reference, loop, loop_hidden, counts, scores, grad_output),
        args.warmup,
        args.iterations,
    )
    grouped_training_ms = _measure_ms(
        _training_step(grouped, grouped_backend, grouped_hidden, counts, scores, grad_output),
        args.warmup,
        args.iterations,
    )

    print(
        f"shape: E={num_local_experts} rows={num_tokens} H={config.hidden_size} "
        f"I={config.moe_intermediate_size}"
    )
    print(
        f"forward: loop={loop_forward_ms:.3f} ms "
        f"te_grouped={grouped_forward_ms:.3f} ms "
        f"speedup={loop_forward_ms / grouped_forward_ms:.2f}x"
    )
    print(
        f"forward+backward: loop={loop_training_ms:.3f} ms "
        f"te_grouped={grouped_training_ms:.3f} ms "
        f"speedup={loop_training_ms / grouped_training_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
