# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-rank PLE row-shard lookup, gradients, and checkpoint schema."""

from __future__ import annotations

import importlib.util
import io
import socket
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F


def _config():
    from megatron.core.models.qwenair import QwenAirTextConfig

    return QwenAirTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        hc_count=4,
        hc_lowrank=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 0),
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=7,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=3,
    )


def _check_lookup_and_gradients(module, indices, full_weight):
    """Compare each rank's variable-size lookup to a summed dense-table oracle."""
    reference_weight = full_weight.detach().clone().requires_grad_()
    actual = module.lookup_indices(indices)
    expected = F.embedding(indices, reference_weight)
    torch.testing.assert_close(actual, expected)
    coefficients = (
        torch.arange(actual.numel(), device=actual.device).reshape(actual.shape).float() + 1
    ) / 17
    coefficients += dist.get_rank() / 11
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    dist.all_reduce(reference_weight.grad, group=module.process_group)
    expected_grad = reference_weight.grad[module.shard_start : module.shard_end]
    torch.testing.assert_close(module.ngram_embedding.weight.grad, expected_grad)
    module.ngram_embedding.weight.grad = None


def _worker(rank: int, port: int, device_type: str, checkpoint_dir: str | None) -> None:
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=2, timeout=timedelta(seconds=90),
    )
    try:
        from megatron.core.models.qwenair import QwenAirForCausalLM
        from megatron.core.models.qwenair.ple import (
            QwenAirNGramEmbedding,
            qwenair_ngram_indices,
            qwenair_ngram_metadata,
        )

        config = _config()
        _, _, _, rows = qwenair_ngram_metadata(config)
        width = config.ple_embed_dim // ((config.ngram_size - 1) * config.heads_per_ngram)
        full_weight = (
            torch.arange(rows * width, dtype=torch.float32, device=device).reshape(rows, width)
            / 113
        )
        module = QwenAirNGramEmbedding(config, 0, 0, process_group=dist.group.WORLD).to(device)
        assert module.rows_per_rank == rows // 2
        with torch.no_grad():
            module.ngram_embedding.weight.copy_(full_weight[module.shard_start : module.shard_end])

        # Only rank 0 owns these rows; rank 1 must handle an empty local
        # embedding request while receiving nonempty return values.
        direct = torch.tensor([0, 1, 2] if rank == 0 else [2, 3], device=device)
        _check_lookup_and_gradients(module, direct, full_weight)

        tokens = torch.tensor([[1, 2, 3]] if rank == 0 else [[3, 4, 5, 6, 7, 8]], device=device)
        hashed = qwenair_ngram_indices(config, tokens)
        _check_lookup_and_gradients(module, hashed, full_weight)
        torch.testing.assert_close(
            module(tokens), F.embedding(hashed, full_weight).flatten(-2)
        )

        local_state = module.state_dict()
        assert local_state["ngram_embedding.weight"].shape == (rows // 2, width)
        saved = io.BytesIO()
        torch.save(local_state, saved)
        saved.seek(0)
        restored = QwenAirNGramEmbedding(config, 0, 0, process_group=dist.group.WORLD).to(device)
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        torch.testing.assert_close(restored.ngram_embedding.weight, module.ngram_embedding.weight)

        descriptor = module.sharded_state_dict(
            prefix="ple_embedding.", metadata={"dp_cp_group": None}
        )
        shard = descriptor[
            "ple_embedding.ngram_embedding.weight"
        ]
        assert shard.global_shape == (rows, width)
        assert shard.local_shape == (rows // 2, width)
        assert shard.global_offset == (rank * rows // 2, 0)
        if checkpoint_dir is not None:
            from megatron.core.dist_checkpointing import load as load_sharded
            from megatron.core.dist_checkpointing import save as save_sharded

            save_sharded(descriptor, checkpoint_dir)
            dist.barrier()
            reloaded = QwenAirNGramEmbedding(
                config, 0, 0, process_group=dist.group.WORLD
            ).to(device)
            loaded_state = load_sharded(
                reloaded.sharded_state_dict(
                    prefix="ple_embedding.", metadata={"dp_cp_group": None}
                ),
                checkpoint_dir,
            )
            reloaded.load_state_dict(
                {
                    name.removeprefix("ple_embedding."): value
                    for name, value in loaded_state.items()
                },
                strict=True,
            )
            torch.testing.assert_close(
                reloaded.ngram_embedding.weight, module.ngram_embedding.weight
            )

        model = QwenAirForCausalLM(config, ple_process_group=dist.group.WORLD).to(device)
        model_shards = model.sharded_state_dict(prefix="qwenair.", metadata={"dp_cp_group": None})
        table_key = "qwenair.model.layers.0.ple.ple_embedding.ngram_embedding.weight"
        assert model_shards[table_key].global_shape == (rows, width)
        assert model_shards[table_key].global_offset == (rank * rows // 2, 0)
        assert model_shards["qwenair.model.embed_tokens.weight"].global_offset == (0, 0)
    finally:
        dist.destroy_process_group()


def _run_two_rank_test(device_type: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    if importlib.util.find_spec("psutil") is None:
        # MCore's torch-distributed checkpoint writer requires psutil. Still
        # check row-shard metadata and strict local state_dict restoration.
        mp.spawn(_worker, args=(port, device_type, None), nprocs=2, join=True)
    else:
        workspace = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory(
            prefix=f"qwenair-ple-{device_type}-", dir=workspace
        ) as checkpoint_root:
            checkpoint_dir = Path(checkpoint_root) / "sharded"
            checkpoint_dir.mkdir()
            mp.spawn(_worker, args=(port, device_type, str(checkpoint_dir)), nprocs=2, join=True)


def test_two_rank_variable_length_ple_lookup_and_checkpoint():
    """Cross-rank requests retain HF outputs, summed grads, and row-shard schema."""
    _run_two_rank_test("cpu")


def test_two_rank_variable_length_ple_lookup_cuda():
    """The same variable all-to-all and backward contract works over NCCL."""
    import pytest

    if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip("Two CUDA devices and NCCL are required")
    _run_two_rank_test("cuda")


if __name__ == "__main__":
    test_two_rank_variable_length_ple_lookup_and_checkpoint()
