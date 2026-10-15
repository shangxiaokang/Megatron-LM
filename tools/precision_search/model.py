# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dependency-free policy, measurement, and export contracts."""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = ("nvfp4", "mxfp8", "bf16")


def write_json(path: Path, value: object) -> None:
    """Atomically publish an artifact, rejecting nonfinite numerical results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Policy:
    """The entire search space: a format assignment to named compute modules."""

    default: str = "nvfp4"
    overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.default not in FORMATS:
            raise ValueError(f"Unknown precision: {self.default}")
        names = [name for name, _ in self.overrides]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate module assignments")
        if any(not name or precision not in FORMATS for name, precision in self.overrides):
            raise ValueError("Invalid module assignment")

    def precision(self, name: str) -> str:
        return dict(self.overrides).get(name, self.default)

    def assign(self, names: list[str], precision: str) -> "Policy":
        assignments = dict(self.overrides)
        for name in names:
            if precision == self.default:
                assignments.pop(name, None)
            else:
                assignments[name] = precision
        return Policy(self.default, tuple(sorted(assignments.items())))

    def as_dict(self) -> dict:
        return {"default": self.default, "overrides": dict(self.overrides)}

    @classmethod
    def from_dict(cls, value: dict) -> "Policy":
        return cls(value["default"], tuple(sorted(value.get("overrides", {}).items())))

    @property
    def key(self) -> str:
        serialized = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def export(self) -> dict:
        """Return JSON/YAML-compatible Megatron TE precision configuration.

        Parameter storage inherits the user's policy. Validation executes the
        candidate's compute precision, including direct forward quantization error.
        Exact module names precede the catch-all matcher.
        """
        configs = {}
        for precision in FORMATS:
            # An already-unquantized context needs no BF16 override. Forcing an
            # extra autocast around every linear adds measurable CPU overhead.
            # The backend's default override_quantized_autocast=True still
            # disables an enclosing FP8/FP4 context for protected BF16 modules.
            recipe = {
                "override_nonquantized_autocast": precision != "bf16",
                "inherit_model_init_context": True,
            }
            if precision == "nvfp4":
                recipe["fp4_quantization_recipe"] = "nvfp4"
            elif precision == "mxfp8":
                recipe["fp8_quantization_recipe"] = "mxfp8"
            configs[precision] = {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": recipe,
                "evaluation_recipe": dict(recipe),
            }
        matchers = {}
        for index, (name, precision) in enumerate(sorted(self.overrides)):
            # fnmatch treats brackets as patterns, so escape literal model names.
            pattern = "".join({"[": "[[]", "*": "[*]", "?": "[?]"}.get(c, c) for c in name)
            matchers[f"module_{index}"] = {
                "config": precision,
                "type": "glob",
                "pattern": pattern,
                "enabled": True,
            }
        matchers["default"] = {
            "config": self.default,
            "type": "glob",
            "pattern": "*",
            "enabled": True,
        }
        return {"configs": configs, "matchers": matchers}


@dataclass
class Result:
    """One actual training branch, with per-batch validation and steady step timing."""

    policy: Policy
    steps: int
    cohort: str
    status: str
    validation: dict[str, list[float]] = field(default_factory=dict)
    step_ms: list[float] = field(default_factory=list)
    modules: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    reason: str = ""
    # Append new measurements to preserve positional construction and old JSON artifacts.
    training: dict[str, list[float]] = field(default_factory=dict)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.step_ms) if self.step_ms else math.inf

    def as_dict(self) -> dict:
        return {**self.__dict__, "policy": self.policy.as_dict()}

    @classmethod
    def from_dict(cls, value: dict) -> "Result":
        value = dict(value)
        value["policy"] = Policy.from_dict(value["policy"])
        return cls(**value)


def vector_error(reference: list[float], candidate: list[float]) -> dict:
    """Normalized error and angle; undefined comparisons never become perfect scores."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Diagnostic samples must have the same nonzero length")
    if not all(math.isfinite(x) for x in reference + candidate):
        raise ValueError("Nonfinite diagnostic samples")
    a2 = sum(x * x for x in reference)
    b2 = sum(x * x for x in candidate)
    error2 = sum((x - y) ** 2 for x, y in zip(reference, candidate))
    cosine = sum(x * y for x, y in zip(reference, candidate)) / max(math.sqrt(a2 * b2), 1e-30)
    if a2 == b2 == 0:
        cosine = 1.0
    return {"relative_l2": math.sqrt(error2 / max(a2, 1e-30)), "cosine": cosine}


def quality_gate(
    candidate: Result,
    reference: Result,
    metric: str,
    tolerance: float,
    *,
    criterion: str = "block_upper_bound",
) -> dict:
    """A conservative paired, block-mean screening bound, not a convergence proof.

    Consecutive validation batches are grouped to reduce sensitivity to correlated
    samples. The final cohort is disjoint from the tuning cohort. Loss is minimized;
    tolerance is relative to the matched BF16 validation loss, never inferred from noise.
    The optional mean criterion changes the acceptance statistic, retaining the
    paired block bound in the report and all matching/finite measurement guards.
    """
    if criterion not in ("block_upper_bound", "mean"):
        raise ValueError(f"Unknown loss criterion: {criterion}")
    decision = {"passed": False, "reason": "incomplete or unmatched trial"}
    if (
        candidate.status != "complete"
        or reference.status != "complete"
        or candidate.steps != reference.steps
        or candidate.cohort != reference.cohort
    ):
        return decision
    a, b = candidate.validation.get(metric, []), reference.validation.get(metric, [])
    if len(a) != len(b) or len(a) < 4 or not all(math.isfinite(x) for x in a + b):
        return {"passed": False, "reason": "missing, nonfinite, or unmatched validation batches"}
    differences = [x - y for x, y in zip(a, b)]
    width = max(1, len(differences) // 4)
    blocks = [statistics.mean(differences[i : i + width]) for i in range(0, len(a), width)]
    mean = statistics.mean(differences)
    se = statistics.stdev(blocks) / math.sqrt(len(blocks))
    upper = mean + 2.353 * se
    scale = max(abs(statistics.mean(b)), 1e-12)
    allowance = tolerance * scale
    return {
        "passed": (upper if criterion == "block_upper_bound" else mean) <= allowance,
        "mean_loss_delta": mean,
        "screening_upper_bound": upper,
        "allowed_loss_delta": allowance,
        "relative_mean_loss_delta": mean / scale,
        "relative_screening_upper_bound": upper / scale,
        "criterion": criterion,
        "reason": "paired validation screening",
    }
