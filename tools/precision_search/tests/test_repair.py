# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for repairing a supplied low-precision training policy.

The objective is independent of search ordering: a sensitive layer has a large
causal loss error until it is restored to BF16. Other layers have small additive
errors, and mixed policies intentionally run more slowly than pure BF16.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.precision_search.model import Policy, Result, write_json
from tools.precision_search.runner import CommandRunner
from tools.precision_search.search import Search, SearchOptions


class SyntheticRunner:
    """Measure an additive loss function with paired training/validation data."""

    def __init__(
        self,
        sensitive=(0,),
        reject_holdout=False,
        training_unrepairable=False,
        missing_training=False,
        forced_bf16=(),
    ):
        self.remaining = 100000
        self.names = [f"decoder.layers.{index}.fc1" for index in range(8)]
        self.sensitive = set(sensitive)
        self.reject_holdout = reject_holdout
        self.training_unrepairable = training_unrepairable
        self.missing_training = missing_training
        self.forced_bf16 = set(forced_bf16)
        self.calls = []

    def run(self, policy, steps, cohort, diagnostic=False):
        self.remaining -= 1
        self.calls.append((policy, steps, cohort, diagnostic))
        all_bf16 = all(policy.precision(name) == "bf16" for name in self.names)
        error = sum(
            (0.06 if index in self.sensitive else 0.00003)
            for index, name in enumerate(self.names)
            if policy.precision(name) != "bf16"
        )
        reference_loss = 3.2 if cohort == "holdout" else 3.0
        validation_error = error
        if self.reject_holdout and cohort == "holdout" and not all_bf16:
            validation_error += 0.06
        training_error = 0.06 if self.training_unrepairable and not all_bf16 else error
        result = Result(
            policy,
            steps,
            cohort,
            "complete",
            validation={
                "lm loss": [
                    reference_loss + validation_error + (0.001 if i % 2 else -0.001)
                    for i in range(8)
                ]
            },
            step_ms=[
                (
                    5.0
                    if all_bf16
                    else 8.0 + sum(policy.precision(name) == "bf16" for name in self.names)
                )
            ]
            * 8,
            modules={
                name: {
                    "type": "TEColumnParallelLinear",
                    "allowed": (
                        ["bf16"] if index in self.forced_bf16 else ["bf16", "mxfp8", "nvfp4"]
                    ),
                }
                for index, name in enumerate(self.names)
            },
            training=(
                {}
                if self.missing_training
                else {
                    "lm loss": [
                        reference_loss + training_error + (0.001 if i % 2 else -0.001)
                        for i in range(steps)
                    ]
                }
            ),
            metadata={
                "ranks": [
                    {
                        "initial_iteration": 180000,
                        "initial_weights_digest": "synthetic-frozen-checkpoint",
                        "seed": 42,
                        "initial_train_samples": 12345,
                        "remaining_train_iters": 1000,
                    }
                ]
            },
        )
        if diagnostic:
            result.diagnostics = {
                f"rank0:{name}": {"output": [1.0, 2.0 + error]} for name in self.names
            }
        return result


def repair_options(**changes):
    values = {
        "strategy": "repair",
        "base_policy": Policy("nvfp4"),
        "quality_criterion": "mean",
        "tolerance": 0.001,
        "min_low_precision_fraction": 0.75,
        "require_mixed": True,
        "min_loss_improvement": 0.0001,
        "repair_steps": 8,
        "short_steps": 16,
        "final_steps": 64,
        "max_trials": 48,
        "metric": "lm loss",
    }
    values.update(changes)
    return SearchOptions(**values)


class RepairSearchTests(unittest.TestCase):
    def assert_not_exported(self, report, directory):
        self.assertEqual(report["status"], "not_verified")
        self.assertNotIn("best_policy", report)
        self.assertFalse((directory / "best_precision.json").exists())

    def test_sensitive_layer_is_repaired_despite_bf16_being_faster(self):
        runner = SyntheticRunner()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options()).run()
            self.assertEqual(report["status"], "verified")
            policy = Policy.from_dict(report["best_policy"])
            self.assertEqual(policy.precision(runner.names[0]), "bf16")
            self.assertEqual(policy.default, "nvfp4")
            low_fraction = sum(policy.precision(name) != "bf16" for name in runner.names) / len(
                runner.names
            )
            self.assertGreaterEqual(low_fraction, 0.75)
            self.assertEqual(report["low_precision_fraction"], low_fraction)
            self.assertLess(report["speedup"], 1.0)
            self.assertTrue((directory / "best_precision.json").is_file())
            self.assertTrue(
                any(
                    p == policy and cohort == "holdout" and steps == 64 and not diagnostic
                    for p, steps, cohort, diagnostic in runner.calls
                )
            )
            accepted = [entry for entry in report["protection_ablation"] if entry["accepted"]]
            self.assertTrue(accepted)
            for entry in accepted:
                self.assertGreaterEqual(entry["validation_improvement_relative"], 0.0001)
                self.assertGreaterEqual(entry["training_improvement_relative"], 0.0001)
                self.assertTrue(entry["significant"])

    def test_coverage_constraint_prevents_bf16_fallback(self):
        runner = SyntheticRunner(sensitive=(0, 1, 2))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options(min_low_precision_fraction=0.8)).run()
            self.assert_not_exported(report, directory)

    def test_fresh_holdout_rejects_tuning_only_success(self):
        runner = SyntheticRunner(reject_holdout=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options()).run()
            self.assertTrue(any(call[2] == "holdout" for call in runner.calls))
            self.assert_not_exported(report, directory)

    def test_validation_success_cannot_hide_training_loss_failure(self):
        runner = SyntheticRunner(training_unrepairable=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options()).run()
            self.assert_not_exported(report, directory)

    def test_missing_training_measurements_are_not_a_success(self):
        runner = SyntheticRunner(missing_training=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options()).run()
            self.assert_not_exported(report, directory)

    def test_unknown_baseline_module_is_rejected(self):
        runner = SyntheticRunner()
        policy = Policy("nvfp4").assign(["decoder.layers.999.fc1"], "bf16")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                Search(runner, Path(temporary), repair_options(base_policy=policy)).run()

    def test_illegal_explicit_baseline_precision_is_rejected(self):
        runner = SyntheticRunner(forced_bf16=(7,))
        policy = Policy("nvfp4").assign([runner.names[7]], "mxfp8")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                Search(runner, Path(temporary), repair_options(base_policy=policy)).run()

    def test_existing_baseline_overrides_are_preserved(self):
        runner = SyntheticRunner()
        baseline = (
            Policy("nvfp4").assign([runner.names[3]], "bf16").assign([runner.names[4]], "mxfp8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = Search(runner, Path(temporary), repair_options(base_policy=baseline)).run()
            self.assertEqual(report["status"], "verified")
            chosen = Policy.from_dict(report["best_policy"])
            self.assertEqual(chosen.precision(runner.names[3]), "bf16")
            self.assertEqual(chosen.precision(runner.names[4]), "mxfp8")
            for policy, _, _, _ in runner.calls:
                if policy.default != "bf16":
                    self.assertEqual(policy.precision(runner.names[3]), "bf16")
                    self.assertIn(policy.precision(runner.names[4]), ("mxfp8", "bf16"))

    def test_forced_bf16_modules_count_against_low_precision_coverage(self):
        runner = SyntheticRunner(sensitive=(), forced_bf16=(7,))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options(min_low_precision_fraction=0.9)).run()
            self.assert_not_exported(report, directory)

    def test_improvement_threshold_requires_a_measured_effect(self):
        runner = SyntheticRunner()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, repair_options(min_loss_improvement=0.03)).run()
            self.assert_not_exported(report, directory)
            self.assertFalse(any(entry["accepted"] for entry in report["protection_ablation"]))

    def test_resumed_measurements_do_not_change_when_tolerance_changes(self):
        synthetic = SyntheticRunner()
        policy = Policy("nvfp4").assign([synthetic.names[3]], "bf16")
        result = synthetic.run(policy, 16, "tuning")
        reference = synthetic.run(Policy("bf16"), 16, "tuning")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            command = ["unused-synthetic-trainer", "--load", "frozen-checkpoint"]
            CommandRunner(command, directory, 60)
            cached = directory / "trials" / f"{policy.key}-16-tuning-d0" / "result.json"
            write_json(cached, result.as_dict())
            original = cached.read_bytes()
            runner = CommandRunner(command, directory, 60, resume=True)
            resumed = runner.run(policy, 16, "tuning")
            strict = Search(runner, directory, repair_options(tolerance=0.001))
            relaxed = Search(runner, directory, repair_options(tolerance=0.03))
            self.assertFalse(strict.gate(resumed, reference)["passed"])
            self.assertTrue(relaxed.gate(resumed, reference)["passed"])
            self.assertEqual(resumed.validation, result.validation)
            self.assertEqual(resumed.training, result.training)
            self.assertEqual(cached.read_bytes(), original)
            self.assertEqual(json.loads(cached.read_text())["training"], result.training)


if __name__ == "__main__":
    unittest.main()
