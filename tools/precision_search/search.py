# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Automatic diagnosis, causal precision ablation, and static-policy selection."""

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from .model import FORMATS, Policy, Result, quality_gate, vector_error, write_json


@dataclass
class SearchOptions:
    """Budget/quality controls; none of these are training hyperparameters."""

    tolerance: float = 0.005
    max_trials: int = 32
    metric: str | None = None
    short_steps: int = 16
    final_steps: int = 64
    strategy: str = "beam"
    base_policy: Policy | None = None
    quality_criterion: str = "block_upper_bound"
    min_low_precision_fraction: float = 0.0
    require_mixed: bool = False
    min_loss_improvement: float = 0.0001
    repair_steps: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Quality tolerance must be finite and nonnegative")
        if self.max_trials < 8 or self.short_steps < 8 or self.final_steps < self.short_steps:
            raise ValueError("Need >=8 trials, >=8 short steps, and final_steps >= short_steps")
        if self.strategy not in ("beam", "repair"):
            raise ValueError("Search strategy must be beam or repair")
        if self.quality_criterion not in ("mean", "block_upper_bound"):
            raise ValueError("Quality criterion must be mean or block_upper_bound")
        if (
            not math.isfinite(self.min_low_precision_fraction)
            or not 0 <= self.min_low_precision_fraction <= 1
        ):
            raise ValueError("Low-precision fraction must be between zero and one")
        if not math.isfinite(self.min_loss_improvement) or self.min_loss_improvement < 0:
            raise ValueError("Minimum loss improvement must be finite and nonnegative")
        if self.repair_steps < 8:
            raise ValueError("Repair screening needs at least eight training steps")
        if self.strategy == "repair" and self.base_policy is None:
            self.base_policy = Policy("mxfp8")
        if self.strategy == "repair" and self.base_policy.default == "bf16":
            raise ValueError("Repair requires a low-precision baseline default")
        if self.strategy == "beam" and (
            self.base_policy is not None or self.min_low_precision_fraction or self.require_mixed
        ):
            raise ValueError("Baseline and coverage constraints require --strategy repair")


def sensitivity(reference: Result, candidates: list[Result]) -> list[dict]:
    """Rank observed network-output/gradient/update error, never use it as a quality gate."""
    measurements = {name: [] for name in reference.modules}
    details = {name: {} for name in reference.modules}
    for candidate in candidates:
        if candidate.status != "complete":
            continue
        for key, original in reference.diagnostics.items():
            name = key.split(":", 1)[1]
            if name not in measurements:
                continue
            quantized = candidate.diagnostics.get(key, {})
            for role, values in original.items():
                if role not in quantized:
                    continue
                try:
                    error = vector_error(values, quantized[role])
                except ValueError:
                    continue
                measurements[name].append(error["relative_l2"] + max(0, 1 - error["cosine"]))
                details[name].setdefault(candidate.policy.default, {}).setdefault(role, []).append(
                    error
                )
    return sorted(
        (
            {
                "name": name,
                "score": statistics.mean(values) if values else None,
                "measurements": details[name],
            }
            for name, values in measurements.items()
        ),
        key=lambda entry: (-(entry["score"] if entry["score"] is not None else 0), entry["name"]),
    )


def module_groups(names: list[str]) -> list[list[str]]:
    """Create automatic, small depth/type groups; the search recursively refines them."""
    types = {}
    for name in names:
        types.setdefault(name.rsplit(".", 1)[-1], []).append(name)
    groups = []
    for modules in types.values():
        ordered = sorted(
            modules,
            key=lambda name: (
                int(match.group(1)) if (match := re.search(r"layers\.(\d+)", name)) else -1
            ),
        )
        width = max(1, math.ceil(len(ordered) / 4))
        groups.extend(ordered[i : i + width] for i in range(0, len(ordered), width))
    return groups


def backend_protection_policies(modules: dict) -> list[Policy]:
    """Seed heterogeneous GEMM backends without adding training hyperparameters.

    Grouped MoE kernels can pay more for quantization/padding than they save in
    matmul time. Measure protecting that complete backend family, independently
    of diagnostic error ranking and depth groups that mix shared/routed experts.
    """
    grouped = [
        name
        for name, descriptor in modules.items()
        if "GroupedLinear" in descriptor.get("type", "")
    ]
    if not grouped or len(grouped) == len(modules):
        return []
    policies = []
    for precision in FORMATS[:-1]:
        protected = set(grouped) | {
            name for name, descriptor in modules.items() if precision not in descriptor["allowed"]
        }
        policy = Policy(precision).assign(sorted(protected), "bf16")
        if any(policy.precision(name) != "bf16" for name in modules):
            policies.append(policy)
    return policies


class Search:
    """Budgeted bidirectional beam search over legal module format assignments."""

    def __init__(self, runner, output: Path, options: SearchOptions) -> None:
        self.runner = runner
        self.output = output
        self.options = options
        self.results: list[Result] = []
        self.seen = set()
        self.started_budget = runner.remaining
        self.metric = options.metric
        self.modules = {}

    def trial(self, policy: Policy, steps: int, cohort: str, diagnostic: bool = False) -> Result:
        key = (policy.key, steps, cohort, diagnostic)
        if key in self.seen:
            return next(
                result for result in self.results if result.metadata.get("search_key") == list(key)
            )
        result = self.runner.run(policy, steps, cohort, diagnostic)
        result.metadata["search_key"] = list(key)
        self.results.append(result)
        self.seen.add(key)
        write_json(self.output / "progress.json", self.report("running"))
        timing = f"{result.median_ms:.2f} ms" if math.isfinite(result.median_ms) else "no timing"
        print(
            f"[{len(self.results)}] {cohort} {steps} steps {policy.key}: {result.status}, {timing}",
            flush=True,
        )
        return result

    def gate(self, candidate: Result, reference: Result) -> dict:
        if self.options.quality_criterion != "block_upper_bound":
            return quality_gate(
                candidate,
                reference,
                self.metric,
                self.options.tolerance,
                criterion=self.options.quality_criterion,
            )
        return quality_gate(candidate, reference, self.metric, self.options.tolerance)

    def can_search(self) -> bool:
        # Reserve trials and wall time for independent final evidence and a BF16 fallback.
        return (
            len(self.results) < self.options.max_trials - 3
            and self.runner.remaining > self.started_budget * 0.35
        )

    def report(self, status: str) -> dict:
        trials = []
        for result in self.results:
            trial = result.as_dict()
            if trial["diagnostics"]:
                # Raw vectors remain in the runner's per-trial result.json. Do
                # not duplicate them in every progress/selection report: large
                # MoE inventories can otherwise produce hundreds of MiB.
                trial["diagnostics"] = {}
                trial["metadata"] = {**trial["metadata"], "diagnostics_in_trial_result": True}
            trials.append(trial)
        return {
            "schema_version": 1,
            "status": status,
            "search_space": "module BF16/MXFP8/NVFP4 compute assignments only",
            "metric": self.metric,
            "relative_loss_tolerance": self.options.tolerance,
            "strategy": self.options.strategy,
            "quality_criterion": self.options.quality_criterion,
            "claim": "best independently verified policy found within the budget",
            "trials": trials,
        }

    def run(self) -> dict:
        if self.options.strategy == "repair":
            from .repair import RepairSearch

            return RepairSearch(self).run()
        diagnostic_steps = max(8, getattr(self.runner, "minimum_steps", 8))
        bf16 = Policy("bf16")
        reference_diagnostic = self.trial(bf16, diagnostic_steps, "tuning", True)
        if reference_diagnostic.status != "complete":
            raise RuntimeError(f"BF16 diagnostic reference failed: {reference_diagnostic.reason}")
        self.modules = reference_diagnostic.modules
        if self.metric is None:
            available = sorted(reference_diagnostic.validation)
            self.metric = "lm loss" if "lm loss" in available else available[0]
        if self.metric not in reference_diagnostic.validation:
            raise ValueError(f"Metric {self.metric!r} is absent from validation")

        diagnoses = []
        for precision in FORMATS[:-1]:
            if any(precision in descriptor["allowed"] for descriptor in self.modules.values()):
                policy = Policy(precision).assign(
                    [name for name, d in self.modules.items() if precision not in d["allowed"]],
                    "bf16",
                )
                diagnoses.append(self.trial(policy, diagnostic_steps, "tuning", True))
        ranking = sensitivity(reference_diagnostic, diagnoses)
        write_json(self.output / "sensitivity.json", ranking)

        remaining_steps = min(
            (
                rank.get("remaining_train_iters", self.options.final_steps)
                for rank in reference_diagnostic.metadata.get("ranks", [])
            ),
            default=self.options.final_steps,
        )
        short = min(max(self.options.short_steps, diagnostic_steps), remaining_steps)
        if short < diagnostic_steps:
            raise RuntimeError("Original training horizon has too few remaining iterations")
        reference = self.trial(bf16, short, "tuning")
        if reference.status != "complete":
            raise RuntimeError(f"BF16 training reference failed: {reference.reason}")
        initial_policies = [result.policy for result in diagnoses]
        candidates = [reference]
        for policy in initial_policies:
            if self.can_search():
                candidates.append(self.trial(policy, short, "tuning"))

        for policy in backend_protection_policies(self.modules):
            if self.can_search():
                candidates.append(self.trial(policy, short, "tuning"))

        priorities = {entry["name"]: index for index, entry in enumerate(ranking)}
        groups = sorted(
            module_groups(list(self.modules)),
            key=lambda group: min(priorities.get(name, len(priorities)) for name in group),
        )
        # Include individual low-ranked modules so diagnostic heuristics cannot permanently exclude them.
        groups += [[entry["name"]] for entry in ranking]
        frontier = list(candidates)
        expanded = set()
        while frontier and self.can_search():
            frontier.sort(
                key=lambda result: (
                    not self.gate(result, reference)["passed"],
                    statistics.mean(result.validation.get(self.metric, [math.inf])),
                    result.median_ms,
                )
            )
            # Prefer speed among feasible policies; retain an infeasible branch for precision recovery.
            passing = [r for r in frontier if self.gate(r, reference)["passed"]]
            failing = [r for r in frontier if not self.gate(r, reference)["passed"]]
            beam = sorted(passing, key=lambda r: r.median_ms)[:2] + failing[:1]
            next_frontier = []
            for parent in beam:
                if parent.status != "complete":
                    continue
                if parent.policy.key in expanded:
                    continue
                expanded.add(parent.policy.key)
                improvements = []
                offset = (len(expanded) - 1) * max(1, len(groups) // 4) % len(groups)
                for group in groups[offset:] + groups[:offset]:
                    if not self.can_search():
                        break
                    for precision in FORMATS:
                        if not self.can_search():
                            break
                        if not all(precision in self.modules[name]["allowed"] for name in group):
                            continue
                        policy = parent.policy.assign(group, precision)
                        if (policy.key, short, "tuning", False) in self.seen:
                            continue
                        result = self.trial(policy, short, "tuning")
                        candidates.append(result)
                        if result.status == "complete":
                            next_frontier.append(result)
                            improvement = statistics.mean(
                                parent.validation.get(self.metric, [math.inf])
                            ) - statistics.mean(result.validation[self.metric])
                            improvements.append(
                                {
                                    "parent": parent.policy.key,
                                    "candidate": policy.key,
                                    "modules": group,
                                    "format": precision,
                                    "loss_improvement": improvement,
                                    "step_ms_delta": result.median_ms - parent.median_ms,
                                }
                            )
                        # Limit expansions of one parent; keep a beam instead of exhausting its neighborhood.
                        if len(next_frontier) >= 4:
                            break
                    if len(next_frontier) >= 4:
                        break
                write_json(self.output / f"ablation-{parent.policy.key}.json", improvements)
                if len(next_frontier) >= 6:
                    break
            frontier = next_frontier

        feasible = sorted(
            [r for r in candidates if self.gate(r, reference)["passed"]],
            key=lambda result: result.median_ms,
        )
        final_steps = min(max(self.options.final_steps, short), remaining_steps)
        final_reference = self.trial(bf16, final_steps, "holdout")
        if final_reference.status != "complete":
            (self.output / "best_precision.json").unlink(missing_ok=True)
            report = self.report("not_verified")
            report["reason"] = (
                "Insufficient budget or failed independent BF16 reference; no policy exported"
            )
            write_json(self.output / "report.json", report)
            return report
        final_candidates = [final_reference]
        for result in feasible:
            if result.policy == bf16 or len(final_candidates) >= 3 or self.runner.remaining <= 0:
                continue
            final_candidates.append(self.trial(result.policy, final_steps, "holdout"))
        verified = [r for r in final_candidates if self.gate(r, final_reference)["passed"]]
        chosen = min(verified, key=lambda r: r.median_ms)
        # Fresh non-diagnostic workers performed final timing and quality measurements.
        policy_path = self.output / "best_precision.json"
        write_json(policy_path, chosen.policy.export())
        report = self.report("verified")
        report.update(
            {
                "best_policy": chosen.policy.as_dict(),
                "config": str(policy_path.resolve()),
                "quality": self.gate(chosen, final_reference),
                "median_step_ms": chosen.median_ms,
                "bf16_median_step_ms": final_reference.median_ms,
                "speedup": final_reference.median_ms / chosen.median_ms,
                "peak_memory_bytes": chosen.metadata.get("peak_memory_bytes"),
                "validation_scope": f"{final_steps} training steps and disjoint held-out validation batches; long-run convergence unverified",
            }
        )
        write_json(self.output / "report.json", report)
        return report
