# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in measurements inside an actual Megatron training process.

Only imported when MEGATRON_PRECISION_SEARCH_TRIAL is set. The normal training
path keeps no hooks, sampler, additional synchronization, or policy controller.
"""

import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

from .model import Policy, write_json
from .runner import TRIAL_ENV


def source_revision(path: Path) -> str:
    """Identify tested source without updating a repository or using the network."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


class Worker:
    """Track a bounded training branch while retaining its original optimizer/schedule."""

    def __init__(self, model, args, spec: dict) -> None:
        import torch
        import transformer_engine
        from transformer_engine.pytorch.quantization import is_mxfp8_available, is_nvfp4_available

        self.torch = torch
        self.model = model
        self.args = args
        self.spec = spec
        self.policy = Policy.from_dict(spec["policy"])
        self.directory = Path(spec["directory"])
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.steps = 0
        self.skipped = 0
        self.step_ms = []
        self.validation = {}
        self.training = {}
        self._previous_validation = {}
        self.diagnostics = {}
        self._sampling_active = False
        self._handles = []
        self._initial_weights = {}
        self.modules = {}
        self.module_objects = {}
        if not getattr(args, "bf16", False):
            raise ValueError(
                "Precision search requires the existing training command to use --bf16"
            )
        if getattr(args, "perform_rl_step", False) or getattr(args, "skip_train", False):
            raise ValueError("Precision search requires an ordinary supervised training branch")
        if getattr(args, "multiple_validation_sets", False) or getattr(
            args, "full_validation", False
        ):
            raise ValueError(
                "This search adapter currently requires a single bounded validation set"
            )
        if getattr(args, "no_load_optim", False) or getattr(args, "no_load_rng", False):
            raise ValueError("Restore optimizer and RNG state for matched training branches")
        allowed = ["bf16"]
        reasons = {}
        for precision, check in (("mxfp8", is_mxfp8_available), ("nvfp4", is_nvfp4_available)):
            supported, reason = check(return_reason=True)
            if supported:
                allowed.append(precision)
            else:
                reasons[precision] = reason
        for chunk in model:
            for module in chunk.modules():
                if not hasattr(module, "_precision_search_name"):
                    continue
                name = module._precision_search_name
                if not name:
                    raise ValueError(
                        "Unnamed TE module cannot be exported as an exact precision assignment"
                    )
                if name in self.modules:
                    raise ValueError(f"Duplicate TE configuration name on rank {self.rank}: {name}")
                quant = module.te_quant_params
                if quant is None:
                    raise ValueError(f"Precision configuration did not reach {name}")
                recipe = quant.training_recipe
                actual = (
                    "nvfp4"
                    if recipe.fp4_quantization_recipe
                    else ("mxfp8" if recipe.fp8_quantization_recipe else "bf16")
                )
                if actual != self.policy.precision(name):
                    raise ValueError(f"Precision mismatch at {name}: {actual} != requested policy")
                self.modules[name] = {
                    "type": type(module).__name__,
                    "allowed": list(allowed),
                    "unsupported_reasons": reasons,
                    "actual_precision": actual,
                }
                self.module_objects[name] = module
                if spec["diagnostic"]:
                    self.diagnostics[name] = {}
                    self._handles.append(module.register_forward_hook(self._forward_hook(name)))
        if not self.modules:
            # Pipeline embedding-only stages are permitted; global inventory is checked by the runner.
            self.modules = {}
        self.initial_iteration = getattr(args, "iteration", 0)
        warmup = max(3, int(getattr(args, "cuda_graph_warmup_steps", 0)) + 2)
        self.warmup = min(warmup, max(3, spec["steps"] // 2))
        if warmup > spec["steps"] - 3:
            raise ValueError("Trial horizon is too short for the original CUDA Graph warmup")
        if spec["diagnostic"] and getattr(args, "cuda_graph_impl", "none") not in (
            "none",
            None,
            "local",
        ):
            # Keep the original graph configuration. Without eager warmup, omit unsafe
            # CPU sampling hooks and let measured training ablations rank the candidates.
            if getattr(args, "cuda_graph_warmup_steps", 0) < 1:
                for handle in self._handles:
                    handle.remove()
                self._handles.clear()
                self.diagnostics.clear()
                self.spec["diagnostic"] = False
        # Sampling effort is automatic and does not change any training hyperparameter.
        args.eval_iters = 16
        self.validation_window = (
            (max(64, spec["steps"]) // args.eval_interval + 2)
            * args.eval_iters
            * args.eval_global_batch_size
        )
        self.initial_validation_samples = args.consumed_valid_samples
        # The holdout cohort advances only validation samples, never training samples or optimizer state.
        if spec["cohort"] == "holdout":
            args.consumed_valid_samples += self.validation_window
        self._initial_digest = self._weight_digest()
        self.metadata = {
            "rank": self.rank,
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": (
                list(torch.cuda.nccl.version())
                if isinstance(torch.cuda.nccl.version(), tuple)
                else torch.cuda.nccl.version()
            ),
            "transformer_engine": transformer_engine.__version__,
            "te_path": str(Path(transformer_engine.__file__).resolve()),
            "te_revision": source_revision(
                Path(transformer_engine.__file__).resolve().parent.parent
            ),
            "megatron_revision": source_revision(Path(__file__).resolve().parents[2]),
            "initial_iteration": self.initial_iteration,
            "remaining_train_iters": max(0, args.train_iters - self.initial_iteration),
            "initial_train_samples": args.consumed_train_samples,
            "validation_start_samples": args.consumed_valid_samples,
            "initial_weights_digest": self._initial_digest,
            "seed": args.seed,
            "warmup_steps": self.warmup,
            "diagnostic_sampling": bool(self._handles),
            "precision_scope": "per-module fprop/dgrad/wgrad together; parameter storage unchanged",
            "NVTE_environment": {
                key: value for key, value in os.environ.items() if key.startswith("NVTE_")
            },
        }
        torch.cuda.reset_peak_memory_stats()

    def _sample(self, tensor) -> list[float]:
        tensor = tensor.detach()
        if hasattr(tensor, "dequantize"):
            tensor = tensor.dequantize()
        flattened = tensor.reshape(-1)
        if not flattened.numel():
            return []
        count = min(256, flattened.numel())
        # Float32 linspace rounds large final offsets up to numel (e.g. a
        # Qwen vocabulary projection), causing an out-of-range CUDA gather.
        # Integer arithmetic also keeps the deterministic endpoints exact.
        positions = self.torch.arange(count, device=flattened.device, dtype=self.torch.int64)
        positions = positions * (flattened.numel() - 1) // max(1, count - 1)
        return flattened.index_select(0, positions).float().cpu().tolist()

    def _weights(self, module) -> list[float]:
        # Sampling, not a substitute optimizer or a searched storage policy.
        values = []
        for name, parameter in module.named_parameters(recurse=False):
            if "weight" in name:
                master = getattr(parameter, "main_param", None)
                if not isinstance(master, self.torch.Tensor) or not master.numel():
                    master = parameter
                values.extend(self._sample(master))
        return values

    def _weight_digest(self) -> str:
        data = {name: self._weights(module) for name, module in self.module_objects.items()}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _forward_hook(self, name: str):
        def hook(module, inputs, output):
            if (
                not self._sampling_active
                or not module.training
                or self.diagnostics[name].get("output")
            ):
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, self.torch.Tensor):
                return
            self.diagnostics[name]["output"] = self._sample(tensor)
            if tensor.requires_grad:
                tensor.register_hook(self._gradient_hook(name, "grad_output"))
            if inputs and isinstance(inputs[0], self.torch.Tensor) and inputs[0].requires_grad:
                inputs[0].register_hook(self._gradient_hook(name, "grad_input"))

        return hook

    def _gradient_hook(self, name: str, role: str):
        def hook(gradient):
            if self._sampling_active and role not in self.diagnostics[name]:
                self.diagnostics[name][role] = self._sample(gradient)

        return hook

    @property
    def done(self) -> bool:
        return self.steps >= self.spec["steps"]

    def wrap_train_step(self, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            self._sampling_active = self.spec["diagnostic"] and self.steps == 0
            if self._sampling_active:
                self._initial_weights = {
                    name: self._weights(module) for name, module in self.module_objects.items()
                }
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            result = function(*args, **kwargs)
            self.torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if self.steps >= self.warmup:
                self.step_ms.append(elapsed)
            # train_step already averages microbatches and, for numerator/count
            # losses, reduces over its data/context parallel group. Only ranks
            # with an output-stage loss return a nonempty dictionary. Do not
            # introduce a different distributed reduction here. Read after the
            # timing interval so CPU measurement transfer is excluded from it.
            self.record_training(result[0])
            self.skipped += int(result[1])
            if self._sampling_active:
                for name, module in self.module_objects.items():
                    after = self._weights(module)
                    before = self._initial_weights[name]
                    if len(after) == len(before):
                        self.diagnostics[name]["update"] = [a - b for a, b in zip(after, before)]
                for handle in self._handles:
                    handle.remove()
                self._handles.clear()
            self._sampling_active = False
            self.steps += 1
            return result

        return measured

    def record_training(self, losses: dict) -> None:
        """Record one reduced loss per training step without reducing rank replicas."""
        for metric, value in losses.items():
            if hasattr(value, "detach"):
                value = value.detach().reshape(-1).cpu().tolist()
            values = list(value) if isinstance(value, (tuple, list)) else [value]
            if len(values) not in (1, 2):
                raise ValueError(f"Invalid training loss shape for {metric}")
            values = [float(item) for item in values]
            if not all(math.isfinite(item) for item in values):
                raise ValueError(f"Nonfinite training loss for {metric}")
            if len(values) == 2:
                if values[1] <= 0:
                    raise ValueError(f"Training loss contains no valid samples/tokens: {metric}")
                loss = values[0] / values[1]
            else:
                loss = values[0]
            if not math.isfinite(loss):
                raise ValueError(f"Nonfinite training loss for {metric}")
            self.training.setdefault(metric, []).append(loss)

    def begin_validation(self, is_test: bool) -> None:
        if not is_test:
            self.validation = {}
            self._previous_validation = {}

    def record_validation(self, running_losses: dict, is_test: bool) -> None:
        if is_test:
            return
        for metric, tensor in running_losses.items():
            numerator, denominator = tensor.detach().float().cpu().tolist()
            previous = self._previous_validation.get(metric, (0, 0))
            count = denominator - previous[1]
            if count <= 0:
                raise ValueError("Validation batch contains no valid samples/tokens")
            self.validation.setdefault(metric, []).append((numerator - previous[0]) / count)
            self._previous_validation[metric] = (numerator, denominator)

    def finish(self) -> None:
        write_json(
            self.directory / f"rank-{self.rank:05d}.json",
            {
                "rank": self.rank,
                "world_size": self.world_size,
                "steps": self.steps,
                "skipped": self.skipped,
                "modules": self.modules,
                "diagnostics": self.diagnostics,
                "validation": self.validation,
                "training": self.training,
                "step_ms": self.step_ms,
                "peak_memory_bytes": self.torch.cuda.max_memory_allocated(),
                "metadata": self.metadata,
            },
        )
        self.torch.distributed.barrier()

    def reserve_validation_samples(self, sizes) -> tuple:
        """Build the same extended validation index in tuning and holdout branches."""
        train, validation, test = sizes
        # A production checkpoint may have consumed many more validation samples
        # with its original eval_iters. Preserve that cursor while extending the
        # index identically for both cohorts, even after automatic bounded sampling.
        return train, validation + self.validation_window + self.initial_validation_samples, test


def initialize(model, args, training_module) -> Worker:
    """Enable measurements once, after checkpoint load and before data iterator creation."""
    spec = json.loads(Path(os.environ[TRIAL_ENV]).read_text())
    worker = Worker(model, args, spec)
    training_module.train_step = worker.wrap_train_step(training_module.train_step)
    return worker
