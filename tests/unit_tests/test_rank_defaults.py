# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep the logging-rank API consumed by Megatron Bridge compatible."""

import logging
import warnings
from unittest.mock import Mock, patch

import pytest

from megatron.core._rank_utils import (
    get_default_log_ranks,
    log_single_rank,
    set_default_log_ranks,
    warn_single_rank,
)


@pytest.fixture
def restore_default_log_ranks():
    """Prevent a configured rank set from leaking into another test."""
    original = get_default_log_ranks()
    yield
    set_default_log_ranks(original)


def test_default_log_rank_is_zero_and_utils_reexports_setter():
    """Keep existing single-rank logging and the Bridge import contract."""
    from megatron.core.utils import set_default_log_ranks as public_setter

    assert get_default_log_ranks() == (0,)
    assert public_setter is set_default_log_ranks


def test_set_default_log_ranks_sorts_and_deduplicates(restore_default_log_ranks):
    set_default_log_ranks([64, 0, 64])
    assert get_default_log_ranks() == (0, 64)


def test_log_single_rank_uses_default_set_and_explicit_override(restore_default_log_ranks):
    set_default_log_ranks([64, 0])
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = True

    with patch("megatron.core._rank_utils.safe_get_rank", return_value=64):
        log_single_rank(logger, logging.INFO, "value=%s", 42)
        log_single_rank(logger, logging.INFO, "not emitted", rank=0)

    logger.log.assert_called_once_with(logging.INFO, "value=%s", 42)


def test_log_single_rank_skips_rank_query_when_level_disabled():
    logger = Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = False

    with patch("megatron.core._rank_utils.safe_get_rank") as rank_query:
        log_single_rank(logger, logging.DEBUG, "message")

    rank_query.assert_not_called()
    logger.log.assert_not_called()


def test_warn_single_rank_uses_default_set_and_explicit_override(restore_default_log_ranks):
    set_default_log_ranks([0, 64])
    with (
        patch("megatron.core._rank_utils.torch.distributed.is_initialized", return_value=True),
        patch("megatron.core._rank_utils.torch.distributed.get_rank", return_value=64),
    ):
        with pytest.warns(UserWarning, match="default rank"):
            warn_single_rank("default rank")
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            warn_single_rank("explicit rank", rank=0)
        assert emitted == []


def test_warn_single_rank_suppresses_undetermined_rank_warning():
    """An import-time warning should not emit an extra rank-probe warning."""
    with (
        patch("megatron.core._rank_utils.torch.distributed.is_initialized", return_value=False),
        patch("megatron.core._rank_utils.safe_get_rank") as rank_query,
    ):
        rank_query.side_effect = lambda: (warnings.warn("rank unknown"), 0)[1]
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            warn_single_rank("target warning")

    assert len(emitted) == 1
    assert str(emitted[0].message) == "target warning"
