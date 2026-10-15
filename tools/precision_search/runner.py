# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Isolated subprocess trials using the user's unchanged training command."""

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from .model import Policy, Result, write_json

TRIAL_ENV = "MEGATRON_PRECISION_SEARCH_TRIAL"


def option_value(command: list[str], option: str) -> str | None:
    """Read an existing one-valued training option without changing it."""
    for index, token in enumerate(command):
        if token == option and index + 1 < len(command):
            return command[index + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return None


def source_fingerprint(command: list[str]) -> str:
    """Invalidate resumed trials when source, working directory, or checkpoint changes."""
    digest = hashlib.sha256(json.dumps([str(Path.cwd()), command]).encode())
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).resolve().parent.glob("*.py"))
    paths += [
        root / "megatron/training/training.py",
        root / "megatron/core/extensions/transformer_engine.py",
        root / "megatron/core/transformer/transformer_block.py",
        root / "megatron/core/models/gpt/gpt_model.py",
    ]
    paths += [Path(token) for token in command if token.endswith(".py") and Path(token).is_file()]
    for path in sorted(set(paths)):
        if path.is_file():
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    checkpoint = option_value(command, "--load")
    if checkpoint and Path(checkpoint).is_dir():
        for path in sorted(Path(checkpoint).rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for key, value in sorted(os.environ.items()):
        if key.startswith("NVTE_"):
            digest.update(f"{key}={value}".encode())
    return digest.hexdigest()


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    """Replace one-valued CLI options without invoking a shell."""
    result = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == option:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {option}")
            index += 2
        elif token.startswith(option + "="):
            index += 1
        else:
            result.append(token)
            index += 1
    return result + [option, value]


def stop_process(process: subprocess.Popen) -> None:
    """Stop the entire torchrun worker group on timeout or interruption."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def merge_shards(directory: Path, policy: Policy, steps: int, cohort: str) -> Result:
    """Require every rank to finish, then merge distributed evidence."""
    shards = [json.loads(path.read_text()) for path in sorted(directory.glob("rank-*.json"))]
    if not shards:
        raise ValueError("Training produced no precision-search measurements; use patched Megatron")
    world_sizes = {shard["world_size"] for shard in shards}
    ranks = {shard["rank"] for shard in shards}
    if len(world_sizes) != 1 or ranks != set(range(next(iter(world_sizes)))):
        raise ValueError("Incomplete or inconsistent distributed rank results")
    if any(shard["steps"] != steps or shard["skipped"] for shard in shards):
        raise ValueError("Training ended early or skipped optimizer updates")
    modules, diagnostics = {}, {}
    lengths = {len(shard["step_ms"]) for shard in shards}
    if len(lengths) != 1 or next(iter(lengths)) < 3:
        raise ValueError("Insufficient steady-state timing samples")
    timing = [max(values) for values in zip(*(s["step_ms"] for s in shards))]
    validation = None
    training = None
    for shard in shards:
        for name, descriptor in shard["modules"].items():
            if name not in modules:
                modules[name] = dict(descriptor)
            else:
                modules[name]["allowed"] = sorted(
                    set(modules[name]["allowed"]) & set(descriptor["allowed"])
                )
        for name, measurements in shard["diagnostics"].items():
            diagnostics[f"rank{shard['rank']}:{name}"] = measurements
        if shard["validation"]:
            if validation is not None and validation != shard["validation"]:
                raise ValueError(
                    "Validation replicas disagree; explicit rank/group reduction required"
                )
            validation = shard["validation"]
        losses = shard.get("training", {})
        if losses:
            if any(
                len(values) != steps or not all(math.isfinite(value) for value in values)
                for values in losses.values()
            ):
                raise ValueError("Missing, nonfinite, or unmatched training steps")
            if training is not None and training != losses:
                raise ValueError(
                    "Training replicas disagree; explicit rank/group reduction required"
                )
            training = losses
    if not modules or not validation:
        raise ValueError("No configurable TE modules or no validation losses")
    if any("training" in shard for shard in shards) and training is None:
        raise ValueError("No training losses on any rank")
    return Result(
        policy,
        steps,
        cohort,
        "complete",
        validation=validation,
        step_ms=timing,
        modules=modules,
        diagnostics=diagnostics,
        metadata={
            "ranks": [s["metadata"] for s in shards],
            "peak_memory_bytes": max(s["peak_memory_bytes"] for s in shards),
        },
        training=training or {},
    )


class CommandRunner:
    """Execute and cache actual Megatron training branches, with one wall-clock budget."""

    def __init__(
        self, command: list[str], output: Path, budget_seconds: float, resume: bool = False
    ) -> None:
        if not command or budget_seconds <= 0:
            raise ValueError("A training command and positive budget are required")
        self.command = command
        self.output = output.resolve()
        self.deadline = time.monotonic() + budget_seconds
        self.output.mkdir(parents=True, exist_ok=True)
        fingerprint = source_fingerprint(command)
        self.minimum_steps = max(
            8, int(option_value(command, "--cuda-graph-warmup-steps") or 0) + 5
        )
        self._initial_states = None
        manifest = self.output / "command.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text())
            if not resume or existing["fingerprint"] != fingerprint:
                raise ValueError(
                    "Output already exists; --resume requires unchanged command, source, and checkpoint"
                )
        write_json(
            manifest, {"command": command, "fingerprint": fingerprint, "cwd": str(Path.cwd())}
        )

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def run(self, policy: Policy, steps: int, cohort: str, diagnostic: bool = False) -> Result:
        directory = self.output / "trials" / f"{policy.key}-{steps}-{cohort}-d{int(diagnostic)}"
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / "result.json"
        if cached.exists():
            prior = Result.from_dict(json.loads(cached.read_text()))
            if prior.status not in ("timed_out", "budget_exhausted"):
                self._check_initial_states(prior)
                return prior
        if self.remaining <= 0:
            return Result(policy, steps, cohort, "budget_exhausted", reason="wall-clock budget")
        for prior_rank in directory.glob("rank-*.json"):
            prior_rank.unlink()
        config = directory / "precision.json"
        write_json(config, policy.export())
        trial = directory / "trial.json"
        write_json(
            trial,
            {
                "directory": str(directory),
                "policy": policy.as_dict(),
                "steps": steps,
                "cohort": cohort,
                "diagnostic": diagnostic,
            },
        )
        command = replace_option(self.command, "--te-precision-config-file", str(config))
        # Prevent trials from writing into the user's checkpoint or logging directories.
        for option in ("--save", "--tensorboard-dir", "--wandb-save-dir", "--config-logger-dir"):
            if any(token == option or token.startswith(option + "=") for token in command):
                command = replace_option(command, option, str(directory / option[2:]))
        environment = dict(os.environ, **{TRIAL_ENV: str(trial)})
        started = time.monotonic()
        process = None
        try:
            with (directory / "training.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                try:
                    code = process.wait(timeout=self.remaining)
                except subprocess.TimeoutExpired:
                    stop_process(process)
                    result = Result(policy, steps, cohort, "timed_out", reason="wall-clock budget")
                else:
                    if code:
                        tail = (directory / "training.log").read_text(errors="replace")[-6000:]
                        result = Result(
                            policy, steps, cohort, "failed", reason=f"exit {code}: {tail}"
                        )
                    else:
                        try:
                            result = merge_shards(directory, policy, steps, cohort)
                        except (ValueError, KeyError) as error:
                            result = Result(policy, steps, cohort, "failed", reason=str(error))
        except BaseException:
            if process is not None:
                stop_process(process)
            raise
        result.metadata["wall_seconds"] = time.monotonic() - started
        result.metadata["command"] = command
        self._check_initial_states(result)
        write_json(cached, result.as_dict())
        return result

    def _check_initial_states(self, result: Result) -> None:
        if result.status != "complete":
            return
        states = [
            (
                rank["initial_iteration"],
                rank["initial_weights_digest"],
                rank["seed"],
                rank["initial_train_samples"],
            )
            for rank in result.metadata["ranks"]
        ]
        if self._initial_states is None:
            self._initial_states = states
        elif states != self._initial_states:
            result.status = "failed"
            result.reason = (
                "Initial model weights, seed, iteration, or training data position changed"
            )
