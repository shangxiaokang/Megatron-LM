# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pure-text Megatron indexed dataset provider for multimodal_dev.

This reuses Megatron-Core's GPTDataset builder and adapts its sample keys to
the multimodal_dev forward path. It is useful for training the Qwen3.5-VL
decoder on plain text data while keeping the same model entry point.
"""

import json
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, print_rank_0
from megatron.training.utils import get_blend_and_blend_per_split


class TextToMultimodalDataset(Dataset):
    """Adapt GPTDataset samples to the multimodal_dev batch schema."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        result = {
            "input_ids": sample["tokens"],
            "labels": sample["labels"],
            "loss_mask": sample["loss_mask"],
        }

        if "attention_mask" in sample:
            result["attention_mask"] = sample["attention_mask"]

        position_ids = sample.get("position_ids")
        if position_ids is not None:
            # Qwen3.5-VL uses MRoPE position IDs with shape [3, S] per sample.
            result["position_ids"] = (
                position_ids.unsqueeze(0).expand(3, -1).contiguous()
            )

        return result


def _core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=(
            args.tensor_model_parallel_size * args.sequence_parallel
        ),
        dynamic_context_parallel=args.dynamic_context_parallel,
        sft_mock_dataset_config_json=args.sft_mock_dataset_config_json,
    )


def _is_dataset_built_on_rank():
    return parallel_state.get_tensor_model_parallel_rank() == 0


def _wrap_dataset(dataset):
    if dataset is None:
        return None
    if isinstance(dataset, list):
        return [TextToMultimodalDataset(ds) for ds in dataset]
    return TextToMultimodalDataset(dataset)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train / validation / test datasets from Megatron indexed text data."""

    args = get_args()
    config = _core_gpt_dataset_config_from_args(args)

    print_rank_0("> building train, validation, and test datasets for text ...")
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        _is_dataset_built_on_rank,
        config,
    ).build()
    print_rank_0("> finished creating text datasets ...")

    return _wrap_dataset(train_ds), _wrap_dataset(valid_ds), _wrap_dataset(test_ds)
