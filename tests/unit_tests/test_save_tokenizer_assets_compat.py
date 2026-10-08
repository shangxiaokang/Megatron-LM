# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tokenizer asset API used by Megatron-Bridge checkpointing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.training.checkpointing import save_tokenizer_assets


def _single_process(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


def test_null_tokenizer_creates_an_empty_asset_directory(tmp_path, monkeypatch):
    _single_process(monkeypatch)

    save_tokenizer_assets(
        object(),
        SimpleNamespace(tokenizer_type="NullTokenizer"),
        str(tmp_path),
    )

    tokenizer_dir = tmp_path / "tokenizer"
    assert tokenizer_dir.is_dir()
    assert list(tokenizer_dir.iterdir()) == []


def test_sentencepiece_asset_is_copied(tmp_path, monkeypatch):
    _single_process(monkeypatch)
    source = tmp_path / "source.model"
    source.write_bytes(b"qwenair-tokenizer")

    save_tokenizer_assets(
        object(),
        SimpleNamespace(
            tokenizer_type="SentencePieceTokenizer",
            tokenizer_model=str(source),
        ),
        str(tmp_path / "checkpoint"),
    )

    assert (
        tmp_path / "checkpoint" / "tokenizer" / "tokenizer.model"
    ).read_bytes() == source.read_bytes()


def test_huggingface_wrapper_uses_save_pretrained(tmp_path, monkeypatch):
    _single_process(monkeypatch)
    inner = Mock(spec=["save_pretrained"])
    tokenizer = SimpleNamespace(_tokenizer=inner)

    save_tokenizer_assets(
        tokenizer,
        SimpleNamespace(tokenizer_type="HuggingFaceTokenizer"),
        str(tmp_path),
    )

    inner.save_pretrained.assert_called_once_with(str(tmp_path / "tokenizer"))


def test_requested_tokenizer_save_failure_is_raised(tmp_path, monkeypatch):
    _single_process(monkeypatch)
    tokenizer = Mock(spec=["save_pretrained"])
    tokenizer.save_pretrained.side_effect = OSError("write failed")

    with pytest.raises(OSError, match="write failed"):
        save_tokenizer_assets(
            tokenizer,
            SimpleNamespace(tokenizer_type="HuggingFaceTokenizer"),
            str(tmp_path),
            raise_on_error=True,
        )
