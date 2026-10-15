# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run all three automatic stages, then independently replay the exported configuration."""

import json
import subprocess
import sys
from pathlib import Path

from examples.precision_search.smoke import training_command
from tools.precision_search.runner import CommandRunner, replace_option
from tools.precision_search.search import Search, SearchOptions


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    runner = CommandRunner(training_command(), output, 2700)
    report = Search(runner, output, SearchOptions(max_trials=16)).run()
    if report["status"] != "verified":
        raise RuntimeError(json.dumps(report))
    completed = [trial for trial in report["trials"] if trial["status"] == "complete"]
    for precision in ("bf16", "mxfp8", "nvfp4"):
        if not any(trial["policy"]["default"] == precision for trial in completed):
            raise AssertionError(f"No successful real training trial for {precision}")
    command = replace_option(training_command(), "--te-precision-config-file", report["config"])
    command = replace_option(command, "--train-iters", "8")
    # This separate deployment smoke checks config consumption without the trial environment.
    # Its short horizon is a test configuration, not a search over learning-rate schedules.
    with (output / "static-replay.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
    # Verify exact-state checkpoint branches and the distributed optimizer adapter.
    checkpoint_command = replace_option(training_command(), "--save", str(output / "checkpoint"))
    checkpoint_command = replace_option(checkpoint_command, "--save-interval", "1000")
    checkpoint_runner = CommandRunner(checkpoint_command, output / "checkpoint-seed", 600)
    from tools.precision_search.model import Policy
    from tools.precision_search.runner import option_value

    initial = checkpoint_runner.run(Policy("bf16"), 8, "tuning")
    if initial.status != "complete":
        raise RuntimeError(f"Checkpoint creation failed: {initial.reason}")
    checkpoint_path = option_value(initial.metadata["command"], "--save")
    resume_command = replace_option(training_command(), "--load", checkpoint_path)
    resume_runner = CommandRunner(resume_command, output / "checkpoint-branches", 600)
    for precision in ("bf16", "mxfp8", "nvfp4"):
        restored = resume_runner.run(Policy(precision), 8, "tuning")
        if restored.status != "complete":
            raise RuntimeError(f"{precision} checkpoint branch failed: {restored.reason}")
        if restored.metadata["ranks"][0]["initial_iteration"] != 8:
            raise AssertionError("Original checkpoint iteration was not restored")
    (output / "checkpoint-acceptance-passed").touch()
    (output / "acceptance-passed").touch()


if __name__ == "__main__":
    main()
