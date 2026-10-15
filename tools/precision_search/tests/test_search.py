# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contract tests; independent of the GPU-initializing repository conftest."""

import fnmatch
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.precision_search.model import Policy, Result, quality_gate, vector_error, write_json
from tools.precision_search.runner import CommandRunner, merge_shards, replace_option
from tools.precision_search.search import (
    Search,
    SearchOptions,
    backend_protection_policies,
    module_groups,
)
from tools.precision_search.worker import Worker


def complete(policy=Policy("bf16"), steps=16, cohort="tuning", loss=3.0):
    return Result(policy, steps, cohort, "complete", {"lm loss": [loss] * 16}, [5.0] * 8)


class DeterministicRunner:
    """Synthetic causal objective: only layer zero requires protection.

    This tests search logic and failure handling, not GPU numerical fidelity.
    """

    def __init__(self, reject_holdout=False, exhaust_final=False):
        self.remaining = 10000
        self.reject_holdout = reject_holdout
        self.exhaust_final = exhaust_final
        self.calls = []
        self.names = ["decoder.layers.0.fc1", "decoder.layers.1.fc1", "decoder.layers.2.fc2"]

    def run(self, policy, steps, cohort, diagnostic=False):
        self.remaining -= 10
        self.calls.append((policy, steps, cohort, diagnostic))
        if self.exhaust_final and cohort == "holdout":
            return Result(policy, steps, cohort, "budget_exhausted")
        loss = 3.0 + (0.2 if policy.precision(self.names[0]) == "nvfp4" else 0)
        if self.reject_holdout and cohort == "holdout" and policy != Policy("bf16"):
            loss += 0.3
        result = complete(policy, steps, cohort, loss)
        result.step_ms = [
            sum({"nvfp4": 1, "mxfp8": 2, "bf16": 4}[policy.precision(n)] for n in self.names)
        ] * 8
        result.modules = {name: {"allowed": ["nvfp4", "mxfp8", "bf16"]} for name in self.names}
        if diagnostic:
            result.diagnostics = {
                "rank0:"
                + name: {"output": [1, 2], "grad_output": [1, 2] if index else [1, 2 + loss - 3]}
                for index, name in enumerate(self.names)
            }
        result.metadata["peak_memory_bytes"] = 1234
        return result


class WorkerSamplingTests(unittest.TestCase):
    def test_checkpoint_validation_cursor_reserves_both_cohorts(self):
        worker = object.__new__(Worker)
        worker.validation_window = 64
        worker.initial_validation_samples = 10000
        self.assertEqual(worker.reserve_validation_samples((100, 32, 10)), (100, 10096, 10))

    def test_distributed_optimizer_unowned_master_parameter(self):
        class Tensor:
            def __init__(self, values):
                self.values = values

            def numel(self):
                return len(self.values)

        worker = object.__new__(Worker)
        worker.torch = SimpleNamespace(Tensor=Tensor)
        worker._sample = lambda tensor: tensor.values
        parameter = Tensor([1, 2])
        module = SimpleNamespace(named_parameters=lambda recurse: [("weight", parameter)])
        for master in (None, Tensor([])):
            parameter.main_param = master
            self.assertEqual(worker._weights(module), [1, 2])
        parameter.main_param = Tensor([3, 4])
        self.assertEqual(worker._weights(module), [3, 4])


class PolicyTests(unittest.TestCase):
    def test_export_exact_names_and_preserves_storage(self):
        name = "decoder.layers.10.experts[2].fc?"
        policy = Policy("nvfp4").assign([name], "mxfp8")
        exported = policy.export()
        first = next(iter(exported["matchers"].values()))
        self.assertTrue(fnmatch.fnmatchcase(name, first["pattern"]))
        self.assertFalse(fnmatch.fnmatchcase("decoder.layers.10.experts2.fcX", first["pattern"]))
        self.assertEqual(list(exported["matchers"])[-1], "default")
        for config in exported["configs"].values():
            self.assertTrue(config["training_recipe"]["inherit_model_init_context"])
            self.assertNotIn("fp8_param", config["training_recipe"])
            self.assertNotIn("fp4_param", config["training_recipe"])
            self.assertTrue(config["evaluation_recipe"]["inherit_model_init_context"])
        self.assertEqual(policy, Policy.from_dict(policy.as_dict()))
        self.assertFalse(
            exported["configs"]["bf16"]["training_recipe"]["override_nonquantized_autocast"]
        )
        for precision in ("mxfp8", "nvfp4"):
            self.assertTrue(
                exported["configs"][precision]["training_recipe"]["override_nonquantized_autocast"]
            )

    def test_invalid_or_duplicate_formats(self):
        for policy in (("fp16", ()), ("bf16", (("a", "bf16"), ("a", "nvfp4")))):
            with self.assertRaises(ValueError):
                Policy(*policy)

    def test_validation_includes_direct_forward_quantization_error(self):
        configs = Policy("nvfp4").export()["configs"]
        self.assertEqual(configs["nvfp4"]["evaluation_recipe"]["fp4_quantization_recipe"], "nvfp4")
        self.assertEqual(configs["mxfp8"]["evaluation_recipe"]["fp8_quantization_recipe"], "mxfp8")
        for precision in ("nvfp4", "mxfp8"):
            self.assertTrue(
                configs[precision]["evaluation_recipe"]["override_nonquantized_autocast"]
            )
        self.assertNotIn("fp4_quantization_recipe", configs["bf16"]["evaluation_recipe"])
        self.assertNotIn("fp8_quantization_recipe", configs["bf16"]["evaluation_recipe"])
        self.assertFalse(configs["bf16"]["evaluation_recipe"]["override_nonquantized_autocast"])
        configs["nvfp4"]["training_recipe"]["fp4_quantization_recipe"] = None
        self.assertEqual(configs["nvfp4"]["evaluation_recipe"]["fp4_quantization_recipe"], "nvfp4")

    def test_assignment_roundtrip(self):
        policy = Policy("nvfp4")
        self.assertEqual(policy.assign(["a"], "bf16").assign(["a"], "nvfp4"), policy)

    def test_relative_error(self):
        self.assertEqual(vector_error([0, 0], [0, 0]), {"relative_l2": 0, "cosine": 1})
        self.assertAlmostEqual(vector_error([1, 0], [0, 1])["relative_l2"], math.sqrt(2))
        with self.assertRaises(ValueError):
            vector_error([1], [float("nan")])


class QualityTests(unittest.TestCase):
    def test_quality_constraint_and_uncertainty(self):
        reference = complete()
        self.assertTrue(quality_gate(complete(loss=3.001), reference, "lm loss", 0.005)["passed"])
        self.assertFalse(quality_gate(complete(loss=3.1), reference, "lm loss", 0.005)["passed"])
        noisy = complete()
        noisy.validation["lm loss"] = [2.5] * 8 + [3.5] * 8
        self.assertFalse(quality_gate(noisy, reference, "lm loss", 0.005)["passed"])

    def test_rejects_mismatched_or_missing_measurements(self):
        reference = complete()
        for candidate in (
            complete(cohort="holdout"),
            complete(steps=64),
            Result(Policy(), 16, "tuning", "failed"),
        ):
            self.assertFalse(quality_gate(candidate, reference, "lm loss", 0.005)["passed"])
        candidate = complete()
        candidate.validation["lm loss"] = [3] * 3
        self.assertFalse(quality_gate(candidate, reference, "lm loss", 0.005)["passed"])


class RunnerTests(unittest.TestCase):
    def test_cli_replacement_preserves_optimizer_and_schedule(self):
        original = [
            "python",
            "pretrain_gpt.py",
            "--lr",
            "0.001",
            "--train-iters",
            "10000",
            "--save=/old",
            "--load",
            "/initial",
        ]
        changed = replace_option(original, "--save", "/isolated")
        self.assertEqual(changed[:-2], original[:-3] + original[-2:])
        self.assertEqual(changed[-2:], ["--save", "/isolated"])
        self.assertEqual(original[5], "10000")

    def test_resume_requires_same_command(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            CommandRunner(["python", "train.py"], directory, 60)
            CommandRunner(["python", "train.py"], directory, 60, resume=True)
            with self.assertRaises(ValueError):
                CommandRunner(["python", "other.py"], directory, 60, resume=True)

    def test_requires_all_rank_artifacts_and_optimizer_updates(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            shard = {
                "rank": 0,
                "world_size": 2,
                "steps": 16,
                "skipped": 0,
                "step_ms": [1, 2, 3],
                "modules": {"a": {"allowed": ["bf16", "mxfp8"]}},
                "diagnostics": {},
                "validation": {"lm loss": [3] * 16},
                "metadata": {},
                "peak_memory_bytes": 100,
            }
            write_json(directory / "rank-00000.json", shard)
            with self.assertRaises(ValueError):
                merge_shards(directory, Policy("bf16"), 16, "tuning")
            shard.update(rank=1, step_ms=[3, 1, 4], peak_memory_bytes=200)
            write_json(directory / "rank-00001.json", shard)
            merged = merge_shards(directory, Policy("bf16"), 16, "tuning")
            self.assertEqual(merged.step_ms, [3, 2, 4])
            self.assertEqual(merged.metadata["peak_memory_bytes"], 200)
            shard["skipped"] = 1
            write_json(directory / "rank-00001.json", shard)
            with self.assertRaises(ValueError):
                merge_shards(directory, Policy("bf16"), 16, "tuning")

    def test_subprocess_failure_keeps_log_and_is_not_a_quality_candidate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = CommandRunner(
                [sys.executable, "-c", "import sys; print('known training error'); sys.exit(7)"],
                Path(temporary),
                20,
            )
            result = runner.run(Policy(), 8, "tuning")
            self.assertEqual(result.status, "failed")
            self.assertIn("known training error", result.reason)
            self.assertTrue(list(Path(temporary).glob("trials/*/training.log")))

    def test_wall_budget_terminates_trial(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            runner = CommandRunner(
                [sys.executable, "-c", "import time; time.sleep(30)"], Path(temporary), 0.1
            )
            start = time.monotonic()
            result = runner.run(Policy(), 8, "tuning")
            self.assertEqual(result.status, "timed_out")
            self.assertLess(time.monotonic() - start, 15)

    def test_changed_checkpoint_invalidates_resume(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            tracker = checkpoint / "latest_checkpointed_iteration.txt"
            tracker.write_text("10")
            command = ["python", "train.py", "--load", str(checkpoint)]
            CommandRunner(command, root / "output", 60)
            tracker.write_text("100")
            with self.assertRaises(ValueError):
                CommandRunner(command, root / "output", 60, resume=True)


class SearchTests(unittest.TestCase):
    def test_backend_protection_respects_capabilities_and_dense_models(self):
        modules = {
            "experts.fc1": {"type": "TEColumnParallelGroupedLinear", "allowed": ["bf16"]},
            "shared.fc1": {"type": "TEColumnParallelLinear", "allowed": ["bf16", "mxfp8"]},
        }
        policies = backend_protection_policies(modules)
        self.assertEqual(policies, [Policy("mxfp8").assign(["experts.fc1"], "bf16")])
        self.assertEqual(backend_protection_policies({"shared.fc1": modules["shared.fc1"]}), [])
        self.assertEqual(backend_protection_policies({"experts.fc1": modules["experts.fc1"]}), [])

    def test_search_protects_expensive_grouped_backend(self):
        class BackendRunner(DeterministicRunner):
            def run(self, policy, steps, cohort, diagnostic=False):
                result = super().run(policy, steps, cohort, diagnostic)
                for index, name in enumerate(self.names):
                    result.modules[name]["type"] = (
                        "TEColumnParallelGroupedLinear" if index < 2 else "TEColumnParallelLinear"
                    )
                result.validation = {
                    "lm loss": [
                        3.2 if any(policy.precision(n) == "nvfp4" for n in self.names) else 3.0
                    ]
                    * 16
                }
                result.step_ms = [
                    sum(
                        (
                            {"bf16": 1, "mxfp8": 4, "nvfp4": 2}
                            if i < 2
                            else {"bf16": 4, "mxfp8": 1, "nvfp4": 0.5}
                        )[policy.precision(name)]
                        for i, name in enumerate(self.names)
                    )
                ] * 8
                return result

        with tempfile.TemporaryDirectory() as temporary:
            runner = BackendRunner()
            report = Search(runner, Path(temporary), SearchOptions(max_trials=16)).run()
            chosen = Policy.from_dict(report["best_policy"])
            self.assertEqual([chosen.precision(n) for n in runner.names], ["bf16", "bf16", "mxfp8"])
            self.assertGreater(report["speedup"], 1)

    def test_report_keeps_raw_diagnostics_in_trial_artifacts(self):
        search = Search(DeterministicRunner(), Path("unused"), SearchOptions())
        result = complete()
        result.diagnostics = {"rank0:fc1": {"output": [1.0] * 10000}}
        search.results = [result]
        report = search.report("running")
        trial = report["trials"][0]
        self.assertEqual(trial["diagnostics"], {})
        self.assertTrue(trial["metadata"]["diagnostics_in_trial_result"])
        self.assertLess(len(json.dumps(report)), 2000)
        self.assertEqual(Result.from_dict(trial).validation, result.validation)
        self.assertEqual(Result.from_dict(trial).step_ms, result.step_ms)
        self.assertEqual(len(result.diagnostics["rank0:fc1"]["output"]), 10000)
        self.assertNotIn("diagnostics_in_trial_result", result.metadata)

    def test_search_finds_protection_and_exports_verified_policy(self):
        runner = DeterministicRunner()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, SearchOptions()).run()
            self.assertEqual(report["status"], "verified")
            chosen = Policy.from_dict(report["best_policy"])
            self.assertNotEqual(chosen.precision(runner.names[0]), "nvfp4")
            self.assertTrue(any(chosen.precision(name) == "nvfp4" for name in runner.names[1:]))
            self.assertGreater(report["speedup"], 1)
            self.assertLessEqual(len(runner.calls), 32)
            self.assertTrue((directory / "best_precision.json").is_file())
            self.assertTrue(any(call[2] == "holdout" and not call[3] for call in runner.calls))

    def test_holdout_failure_automatically_falls_back(self):
        runner = DeterministicRunner(reject_holdout=True)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            report = Search(runner, Path(temporary), SearchOptions(max_trials=16)).run()
            self.assertEqual(report["best_policy"], Policy("bf16").as_dict())

    def test_insufficient_budget_does_not_export_unverified_policy(self):
        runner = DeterministicRunner(exhaust_final=True)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            report = Search(runner, directory, SearchOptions(max_trials=8)).run()
            self.assertEqual(report["status"], "not_verified")
            self.assertFalse((directory / "best_precision.json").exists())

    def test_grouping_covers_every_module(self):
        names = [f"decoder.layers.{i}.fc1" for i in range(24)] + ["output_layer"]
        groups = module_groups(names)
        self.assertCountEqual([name for group in groups for name in group], names)


if __name__ == "__main__":
    unittest.main()
