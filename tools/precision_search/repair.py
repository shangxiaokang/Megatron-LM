# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Measure BF16 restoration benefits starting from a supplied low-precision policy."""

import math
import re
import statistics
from collections import Counter

from .model import Policy, Result, quality_gate, write_json


def protection_groups(modules: dict) -> list[list[str]]:
    """Partition by module family, operator and depth without mixing MoE backends."""
    families = {}
    for name, descriptor in modules.items():
        if ".mlp.experts." in name or "GroupedLinear" in descriptor.get("type", ""):
            family = "routed_experts"
        elif ".shared_experts." in name:
            family = "shared_experts"
        elif ".self_attention." in name:
            family = "attention"
        else:
            family = "dense_or_output"
        families.setdefault((family, name.rsplit(".", 1)[-1]), []).append(name)
    groups = []
    for names in families.values():
        names.sort(
            key=lambda name: (
                int(match.group(1)) if (match := re.search(r"layers\.(\d+)", name)) else -1,
                name,
            )
        )
        width = max(1, math.ceil(len(names) / 4))
        groups.extend(names[index : index + width] for index in range(0, len(names), width))
    return groups


class RepairSearch:
    """Greedy paired ablations, followed by fresh tuning and independent holdout."""

    def __init__(self, controller) -> None:
        self.search = controller
        self.options = controller.options
        self.ablation = []
        self.promotions = []
        self.candidates = []
        self.forced_bf16 = []
        self.baseline = None

    def coverage(self, policy: Policy) -> dict:
        counts = Counter(policy.precision(name) for name in self.search.modules)
        total = len(self.search.modules)
        low = counts["mxfp8"] + counts["nvfp4"]
        return {
            "precision_module_counts": dict(counts),
            "low_precision_fraction": low / total if total else 0.0,
            "coverage_basis": "configurable module count, including forced BF16; not FLOPs",
            "mixed": bool(low and counts["bf16"]),
        }

    def permitted(self, policy: Policy, *, final: bool = True) -> bool:
        coverage = self.coverage(policy)
        return (
            coverage["low_precision_fraction"] > 0
            and (
                coverage["low_precision_fraction"] + 1e-12
                >= self.options.min_low_precision_fraction
            )
            and (not final or not self.options.require_mixed or coverage["mixed"])
        )

    def selection_key(self, result: Result):
        """Prefer fewer additional BF16 modules, then lower loss and step time."""
        restored = sum(
            result.policy.precision(name) == "bf16" and self.baseline.precision(name) != "bf16"
            for name in self.search.modules
        )
        return (restored, statistics.mean(result.validation[self.search.metric]), result.median_ms)

    def gates(self, candidate: Result, reference: Result) -> dict:
        validation = self.search.gate(candidate, reference)
        metric = self.search.metric
        candidate_values = candidate.training.get(metric, [])
        reference_values = reference.training.get(metric, [])
        if len(candidate_values) != candidate.steps or len(reference_values) != reference.steps:
            training = {"passed": False, "reason": "missing or incomplete training loss sequence"}
        else:
            training = quality_gate(
                Result(
                    candidate.policy,
                    candidate.steps,
                    candidate.cohort,
                    candidate.status,
                    validation={metric: candidate_values},
                ),
                Result(
                    reference.policy,
                    reference.steps,
                    reference.cohort,
                    reference.status,
                    validation={metric: reference_values},
                ),
                metric,
                self.options.tolerance,
                criterion=self.options.quality_criterion,
            )
        return {
            "passed": validation["passed"] and training["passed"],
            "validation": validation,
            "training": training,
        }

    def report(self, status: str, reason: str = "") -> dict:
        report = self.search.report(status)
        report.update(
            {
                "base_policy": self.options.base_policy.as_dict(),
                "effective_base_policy": self.baseline.as_dict() if self.baseline else None,
                "forced_bf16_modules": self.forced_bf16,
                "min_low_precision_fraction": self.options.min_low_precision_fraction,
                "require_mixed": self.options.require_mixed,
                "min_loss_improvement_relative_to_bf16": self.options.min_loss_improvement,
                "protection_ablation": self.ablation,
                "accepted_promotions": self.promotions,
                "selection_objective": "fewest added BF16 modules among loss-improving passing policies, then lower loss and step time; BF16 reference excluded",
                "claim": "best independently verified eligible policy found within the budget; not a global optimum",
            }
        )
        if reason:
            report["reason"] = reason
        return report

    def publish(self, status: str, reason: str = "") -> dict:
        if status != "verified":
            (self.search.output / "best_precision.json").unlink(missing_ok=True)
        report = self.report(status, reason)
        write_json(self.search.output / "report.json", report)
        return report

    def can_screen(self) -> bool:
        # Reserve short confirmation and fresh BF16, original baseline and up
        # to two repaired holdout branches. No diagnostic format sweeps.
        return len(self.search.results) < self.options.max_trials - 7 and (
            self.search.runner.remaining > self.search.started_budget * 0.50
        )

    def benefit(self, parent: Result, candidate: Result, reference: Result) -> tuple[float, float]:
        metric = self.search.metric
        if candidate.status != "complete" or parent.status != "complete":
            return -math.inf, -math.inf
        if candidate.steps != parent.steps or candidate.cohort != parent.cohort:
            return -math.inf, -math.inf
        differences = []
        for candidate_values, parent_values, reference_values, expected in (
            (
                candidate.validation.get(metric, []),
                parent.validation.get(metric, []),
                reference.validation.get(metric, []),
                None,
            ),
            (
                candidate.training.get(metric, []),
                parent.training.get(metric, []),
                reference.training.get(metric, []),
                candidate.steps,
            ),
        ):
            if (
                len(candidate_values) != len(parent_values)
                or len(parent_values) != len(reference_values)
                or len(candidate_values) < 4
                or (expected is not None and len(candidate_values) != expected)
                or not all(
                    math.isfinite(value)
                    for value in candidate_values + parent_values + reference_values
                )
            ):
                return -math.inf, -math.inf
            paired = [old - new for old, new in zip(parent_values, candidate_values)]
            differences.append(
                statistics.mean(paired) / max(abs(statistics.mean(reference_values)), 1e-12)
            )
        return tuple(differences)

    def probe(
        self, parent: Result, group: list[str], reference: Result, round_index: int, stage: str
    ):
        changed = [name for name in group if parent.policy.precision(name) != "bf16"]
        if not changed:
            return None
        policy = parent.policy.assign(changed, "bf16")
        if not self.permitted(policy, final=False):
            return None
        key = (policy.key, parent.steps, "tuning", False)
        if key not in self.search.seen and not self.can_screen():
            return None
        candidate = self.search.trial(policy, parent.steps, "tuning")
        validation, training = self.benefit(parent, candidate, reference)
        significant = min(validation, training) > 0 and (
            min(validation, training) + 1e-12 >= self.options.min_loss_improvement
        )
        entry = {
            "parent": parent.policy.key,
            "candidate": policy.key,
            "modules": changed,
            "round": round_index,
            "stage": stage,
            "status": candidate.status,
            "validation_improvement_relative": validation if math.isfinite(validation) else None,
            "training_improvement_relative": training if math.isfinite(training) else None,
            "score": min(validation, training) if significant else None,
            "significant": significant,
            "accepted": False,
            "step_ms_delta": (
                candidate.median_ms - parent.median_ms if candidate.status == "complete" else None
            ),
            **self.coverage(policy),
        }
        self.ablation.append(entry)
        if significant:
            self.candidates.append(candidate)
        write_json(self.search.output / "protection-ablation.json", self.ablation)
        return (candidate, entry) if significant else None

    def run(self) -> dict:
        bf16 = Policy("bf16")
        steps = max(self.options.repair_steps, getattr(self.search.runner, "minimum_steps", 8))
        reference = self.search.trial(bf16, steps, "tuning")
        if reference.status != "complete":
            return self.publish("not_verified", "BF16 screening reference did not complete")
        self.search.modules = reference.modules
        if not self.search.modules:
            raise ValueError("No configurable modules in BF16 reference")
        if self.search.metric is None:
            available = sorted(reference.validation)
            self.search.metric = "lm loss" if "lm loss" in available else available[0]
        if not self.gates(reference, reference)["passed"]:
            return self.publish(
                "not_verified", "BF16 reference lacks matched finite training and validation loss"
            )

        baseline = self.options.base_policy
        unknown = set(dict(baseline.overrides)) - set(self.search.modules)
        if unknown:
            raise ValueError(f"Unknown baseline module names: {sorted(unknown)}")
        for name, descriptor in self.search.modules.items():
            if baseline.precision(name) not in descriptor["allowed"]:
                if name in dict(baseline.overrides):
                    raise ValueError(f"Explicit baseline precision is unsupported for {name}")
                if "bf16" not in descriptor["allowed"]:
                    raise ValueError(f"No supported baseline precision for {name}")
                self.forced_bf16.append(name)
        self.baseline = baseline.assign(self.forced_bf16, "bf16")
        if not self.permitted(self.baseline, final=False):
            return self.publish("not_verified", "Baseline cannot satisfy low-precision coverage")

        parent = self.search.trial(self.baseline, steps, "tuning")
        if parent.status != "complete":
            return self.publish(
                "not_verified", "Low-precision baseline did not complete; no repair evidence"
            )
        self.candidates.append(parent)
        groups = protection_groups(self.search.modules)
        round_index = 0
        while self.can_screen():
            if (
                round_index
                and self.permitted(parent.policy)
                and self.gates(parent, reference)["passed"]
            ):
                break
            round_index += 1
            measured = []
            for group in groups:
                if not self.can_screen():
                    break
                # A coarse restoration exceeding the BF16 allowance is replaced
                # by individual probes so a sensitive singleton is not hidden.
                available = [n for n in group if parent.policy.precision(n) != "bf16"]
                trials = [available]
                if available and not self.permitted(
                    parent.policy.assign(available, "bf16"), final=False
                ):
                    trials = [[name] for name in available]
                for names in trials:
                    result = self.probe(parent, names, reference, round_index, "group")
                    if result:
                        measured.append(result)
                    if not self.can_screen():
                        break
            if not measured:
                break
            winner = max(
                measured,
                key=lambda pair: (pair[1]["score"], -len(pair[1]["modules"]), -pair[0].median_ms),
            )
            # Refine the most useful coarse group before keeping all its modules
            # in BF16; compare every child against the same parent, never against
            # a partially modified or continued training branch.
            if len(winner[1]["modules"]) > 1:
                refined = []
                for name in winner[1]["modules"]:
                    result = self.probe(parent, [name], reference, round_index, "singleton")
                    if result:
                        refined.append(result)
                sufficient = [pair for pair in refined if self.gates(pair[0], reference)["passed"]]
                if sufficient:
                    winner = min(sufficient, key=lambda pair: self.selection_key(pair[0]))
                elif refined:
                    single = max(refined, key=lambda pair: pair[1]["score"])
                    if single[1]["score"] >= winner[1]["score"] * 0.8:
                        winner = single
            parent, entry = winner
            entry["accepted"] = True
            self.promotions.append({"policy": parent.policy.as_dict(), **entry})
            write_json(self.search.output / "protection-ablation.json", self.ablation)
            write_json(self.search.output / "repair-progress.json", self.report("running"))

        unique = {result.policy.key: result for result in self.candidates}
        screened = sorted(
            [
                result
                for result in unique.values()
                if self.permitted(result.policy) and self.gates(result, reference)["passed"]
            ],
            key=self.selection_key,
        )
        improved = [result for result in screened if result.policy != self.baseline]
        if improved:
            screened = improved
        if not screened:
            return self.publish(
                "not_verified",
                "No measured mixed policy meets loss and low-precision coverage constraints within budget",
            )

        remaining_steps = min(
            (
                rank.get("remaining_train_iters", self.options.final_steps)
                for rank in reference.metadata.get("ranks", [])
            ),
            default=self.options.final_steps,
        )
        short = min(max(self.options.short_steps, steps), remaining_steps)
        final = min(max(self.options.final_steps, short), remaining_steps)
        if short < steps:
            return self.publish(
                "not_verified", "Original training horizon is too short for confirmation"
            )
        tuning_reference = self.search.trial(bf16, short, "tuning")
        tuned = []
        # Lock the limited confirmation cohort using tuning evidence. Failed
        # holdout outcomes cannot trigger more test-set driven policy search.
        for candidate in screened[:2]:
            result = self.search.trial(candidate.policy, short, "tuning")
            if self.gates(result, tuning_reference)["passed"]:
                tuned.append(result)
        if not tuned:
            return self.publish(
                "not_verified", "No repaired policy passed longer tuning confirmation"
            )
        write_json(
            self.search.output / "locked-holdout-policies.json", [r.policy.as_dict() for r in tuned]
        )
        final_reference = self.search.trial(bf16, final, "holdout")
        if final_reference.status != "complete":
            return self.publish("not_verified", "Independent BF16 holdout did not complete")
        final_baseline = self.search.trial(self.baseline, final, "holdout")
        if final_baseline.status != "complete":
            return self.publish(
                "not_verified", "Original low-precision baseline holdout did not complete"
            )
        verified = []
        holdout_benefits = {}
        for candidate in tuned:
            result = self.search.trial(candidate.policy, final, "holdout")
            validation, training = self.benefit(final_baseline, result, final_reference)
            improves_baseline = result.policy == self.baseline or (
                min(validation, training) > 0
                and min(validation, training) + 1e-12 >= self.options.min_loss_improvement
            )
            if self.gates(result, final_reference)["passed"] and improves_baseline:
                verified.append(result)
                holdout_benefits[result.policy.key] = {
                    "validation_improvement_relative": validation,
                    "training_improvement_relative": training,
                    "reference_scale": "BF16 mean loss",
                }
        if not verified:
            return self.publish(
                "not_verified",
                "No repaired policy passed independent training and validation loss gates",
            )
        chosen = min(verified, key=self.selection_key)
        path = self.search.output / "best_precision.json"
        write_json(path, chosen.policy.export())
        report = self.report("verified")
        report.update(
            {
                "best_policy": chosen.policy.as_dict(),
                "config": str(path.resolve()),
                "quality": self.gates(chosen, final_reference),
                "original_baseline_quality": self.gates(final_baseline, final_reference),
                "holdout_improvement_vs_original_baseline": holdout_benefits[chosen.policy.key],
                "median_step_ms": chosen.median_ms,
                "bf16_median_step_ms": final_reference.median_ms,
                "speedup": final_reference.median_ms / chosen.median_ms,
                "peak_memory_bytes": chosen.metadata.get("peak_memory_bytes"),
                "validation_scope": f"{final} training steps and disjoint held-out validation batches; long-run convergence unverified",
                **self.coverage(chosen.policy),
            }
        )
        selected_path = []
        policy_key = chosen.policy.key
        while policy_key != self.baseline.key:
            entry = next(
                (
                    e
                    for e in reversed(self.ablation)
                    if e["candidate"] == policy_key and e["significant"]
                ),
                None,
            )
            if entry is None:
                raise RuntimeError("Selected policy lacks measured restoration provenance")
            selected_path.append(entry)
            policy_key = entry["parent"]
        report["selected_protection_ablation"] = list(reversed(selected_path))
        write_json(self.search.output / "report.json", report)
        return report
