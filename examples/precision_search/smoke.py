# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A small, real Megatron GPT branch for Blackwell search integration acceptance."""

import sys
from pathlib import Path

from tools.precision_search.model import Policy
from tools.precision_search.runner import CommandRunner


def training_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=1",
        "--master-port=29547",
        "pretrain_gpt.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "512",
        "--num-attention-heads",
        "8",
        "--ffn-hidden-size",
        "1536",
        "--seq-length",
        "512",
        "--max-position-embeddings",
        "512",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "2",
        "--train-iters",
        "256",
        "--eval-iters",
        "16",
        "--eval-interval",
        "1000000",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "4096",
        "--num-workers",
        "0",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        "unfused",
        "--no-gradient-accumulation-fusion",
        "--use-distributed-optimizer",
        "--lr",
        "0.0001",
        "--lr-decay-style",
        "constant",
        "--log-interval",
        "8",
    ]


if __name__ == "__main__":
    output = Path(sys.argv[1]).resolve()
    runner = CommandRunner(training_command(), output, 600)
    result = runner.run(Policy("bf16"), 8, "tuning", True)
    print(result.status, result.reason, flush=True)
    if result.status != "complete":
        raise SystemExit(1)
    print("Configurable modules:", sorted(result.modules))
    print("Validation metrics:", {key: len(values) for key, values in result.validation.items()})
    print("Median step ms:", result.median_ms)
