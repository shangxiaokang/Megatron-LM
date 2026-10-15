# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-B200 DP/TP/PP policy and measurement acceptance."""

import subprocess
import sys
from pathlib import Path

from examples.precision_search.smoke import training_command
from tools.precision_search.model import Policy
from tools.precision_search.runner import CommandRunner, option_value, replace_option


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    for mode in ("data", "tensor", "pipeline"):
        command = [
            "--nproc-per-node=2" if token == "--nproc-per-node=1" else token
            for token in training_command()
        ]
        command = [
            "--master-port=29549" if token == "--master-port=29547" else token for token in command
        ]
        if mode != "data":
            command = replace_option(command, f"--{mode}-model-parallel-size", "2")
        runner = CommandRunner(command, output / mode, 900)
        protected = "decoder.layers.1.mlp.linear_fc1"
        policies = [Policy(precision) for precision in ("bf16", "mxfp8", "nvfp4")]
        policies.append(Policy("nvfp4").assign([protected], "bf16"))
        for policy in policies:
            result = runner.run(policy, 8, "tuning", True)
            if result.status != "complete":
                raise RuntimeError(f"{mode}/{policy.as_dict()} failed: {result.reason}")
            if len(result.metadata["ranks"]) != 2:
                raise AssertionError("Missing distributed rank measurements")
            if len(result.modules) != 8 or protected not in result.modules:
                raise AssertionError("Global layer inventory is incomplete or aliases PP stages")
            if result.modules[protected]["actual_precision"] != policy.precision(protected):
                raise AssertionError("Per-layer protection did not reach the second global layer")
            print(mode, policy.as_dict(), len(result.modules), result.median_ms, flush=True)
        if mode == "pipeline":
            config = option_value(result.metadata["command"], "--te-precision-config-file")
            replay = replace_option(command, "--te-precision-config-file", config)
            replay = replace_option(replay, "--train-iters", "8")
            with (output / "pipeline-static-replay.log").open("w") as log:
                subprocess.run(replay, stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "distributed-acceptance-passed").touch()


if __name__ == "__main__":
    main()
