# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for reduced training losses and optional mean acceptance."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.precision_search.model import Policy, Result, quality_gate
from tools.precision_search.runner import merge_shards
from tools.precision_search.worker import Worker


def complete(losses, *, steps=8, cohort="tuning"):
    return Result(Policy("bf16"), steps, cohort, "complete", {"lm loss": losses})


def shard(rank, training=None, *, include_training=True):
    value = {
        "rank": rank,
        "world_size": 2,
        "steps": 8,
        "skipped": 0,
        "step_ms": [1, 2, 3],
        "modules": {"decoder.layers.0.fc1": {"allowed": ["bf16", "mxfp8"]}},
        "diagnostics": {},
        "validation": {"lm loss": [3.0] * 8},
        "metadata": {},
        "peak_memory_bytes": 100,
    }
    if include_training:
        value["training"] = {} if training is None else training
    return value


class LossTensor:
    """Only the tensor-to-CPU interface used by training measurements."""

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def reshape(self, size):
        if size != -1:
            raise AssertionError("Loss measurements must flatten before scalar conversion")
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class ResultTrainingTests(unittest.TestCase):
    def test_old_positional_and_json_contracts(self):
        result = Result(Policy(), 8, "tuning", "failed", {}, [], {}, {}, {}, "old reason")
        self.assertEqual(result.reason, "old reason")
        self.assertEqual(result.training, {})
        old = result.as_dict()
        old.pop("training")
        self.assertEqual(Result.from_dict(old), result)

    def test_training_measurements_round_trip(self):
        result = complete([3.0] * 8)
        result.training = {"lm loss": [2.5] * 8}
        self.assertEqual(Result.from_dict(result.as_dict()), result)


class WorkerTrainingTests(unittest.TestCase):
    def setUp(self):
        self.worker = object.__new__(Worker)
        self.worker.training = {}

    def test_scalars_and_numerator_count_pairs(self):
        for value in (3.0, [3.0], (12.0, 4.0), LossTensor([3.0]), LossTensor([12.0, 4.0])):
            self.worker.record_training({"lm loss": value})
        self.assertEqual(self.worker.training, {"lm loss": [3.0] * 5})

    def test_empty_dictionary_is_permitted_on_non_output_stages(self):
        self.worker.record_training({})
        self.assertEqual(self.worker.training, {})

    def test_nonfinite_loss_and_denominator_are_rejected(self):
        for value in (math.nan, math.inf, [1.0, math.nan], [math.inf, 1.0], [1e308, 1e-308]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.worker.record_training({"lm loss": value})

    def test_zero_count_and_unsupported_shapes_are_rejected(self):
        for value in ([1.0, 0.0], [1.0, -1.0], [], [1.0, 2.0, 3.0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.worker.record_training({"lm loss": value})

    def test_train_step_wrapper_records_returned_loss_without_changing_result(self):
        worker = self.worker
        worker.spec = {"diagnostic": False}
        worker.steps = 0
        worker.warmup = 0
        worker.step_ms = []
        worker.skipped = 0
        worker.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        returned = ({"lm loss": LossTensor([12.0, 4.0])}, 0, False, False, 0)
        measured = worker.wrap_train_step(lambda: returned)
        self.assertIs(measured(), returned)
        self.assertEqual(worker.training, {"lm loss": [3.0]})
        self.assertEqual(worker.steps, 1)
        self.assertEqual(len(worker.step_ms), 1)


class MergeTrainingTests(unittest.TestCase):
    def merge(self, first, second):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            # Deliberately allow NaN in a fixture to check ingestion guards.
            for value in (first, second):
                (directory / f"rank-{value['rank']:05d}.json").write_text(
                    json.dumps(value), encoding="utf-8"
                )
            return merge_shards(directory, Policy("bf16"), 8, "tuning")

    def test_matching_replicas_keep_one_reduced_sequence(self):
        losses = {"lm loss": [3.0] * 8}
        merged = self.merge(shard(0, losses), shard(1, losses))
        self.assertEqual(merged.training, losses)
        self.assertEqual(len(merged.training["lm loss"]), 8)

    def test_non_output_stage_may_have_no_training_losses(self):
        losses = {"lm loss": [3.0] * 8}
        self.assertEqual(self.merge(shard(0), shard(1, losses)).training, losses)

    def test_old_shards_without_training_are_backward_compatible(self):
        merged = self.merge(shard(0, include_training=False), shard(1, include_training=False))
        self.assertEqual(merged.training, {})

    def test_new_shards_with_all_training_losses_missing_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No training losses"):
            self.merge(shard(0), shard(1))

    def test_different_training_replicas_need_explicit_reduction(self):
        with self.assertRaisesRegex(ValueError, "Training replicas disagree"):
            self.merge(shard(0, {"lm loss": [3.0] * 8}), shard(1, {"lm loss": [3.1] * 8}))

    def test_missing_or_nonfinite_training_steps_are_rejected(self):
        for values in ([3.0] * 7, [3.0] * 9, [], [math.nan] * 8, [math.inf] * 8):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "training steps"):
                self.merge(shard(0), shard(1, {"lm loss": values}))


class MeanQualityGateTests(unittest.TestCase):
    def test_mean_and_block_bound_are_distinct_and_both_are_reported(self):
        reference = complete([3.0] * 8)
        candidate = complete([2.7, 2.7, 3.0, 3.0, 3.0, 3.0, 3.3, 3.3])
        mean = quality_gate(candidate, reference, "lm loss", 0.001, criterion="mean")
        bound = quality_gate(candidate, reference, "lm loss", 0.001)
        self.assertTrue(mean["passed"])
        self.assertFalse(bound["passed"])
        self.assertAlmostEqual(mean["relative_mean_loss_delta"], 0.0)
        self.assertGreater(mean["relative_screening_upper_bound"], 0.001)
        self.assertEqual(mean["screening_upper_bound"], bound["screening_upper_bound"])
        self.assertEqual(bound["criterion"], "block_upper_bound")

    def test_mean_criterion_retains_complete_step_and_cohort_guards(self):
        reference = complete([3.0] * 8)
        candidates = [complete([3.0] * 8, steps=9), complete([3.0] * 8, cohort="holdout")]
        failed = complete([3.0] * 8)
        failed.status = "failed"
        candidates.append(failed)
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    quality_gate(candidate, reference, "lm loss", 0.001, criterion="mean")["passed"]
                )

    def test_mean_criterion_rejects_missing_nonfinite_or_unmatched_batches(self):
        reference = complete([3.0] * 8)
        for values in ([], [3.0] * 3, [3.0] * 7, [math.nan] * 8, [math.inf] * 8):
            with self.subTest(values=values):
                self.assertFalse(
                    quality_gate(complete(values), reference, "lm loss", 0.001, criterion="mean")[
                        "passed"
                    ]
                )
        missing = complete([3.0] * 8)
        missing.validation = {}
        self.assertFalse(
            quality_gate(missing, reference, "lm loss", 0.001, criterion="mean")["passed"]
        )

    def test_unknown_criterion_is_rejected(self):
        reference = complete([3.0] * 8)
        with self.assertRaisesRegex(ValueError, "Unknown loss criterion"):
            quality_gate(reference, reference, "lm loss", 0.001, criterion="invalid")


if __name__ == "__main__":
    unittest.main()
