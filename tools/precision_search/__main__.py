# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run: python -m tools.precision_search --output results -- <training command>."""

import argparse
import json
import sys
from pathlib import Path

from .model import Policy
from .runner import CommandRunner
from .search import Search, SearchOptions


def main(argv: list[str] | None = None) -> int:
    """Search compute precision only, with no manual stage configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, default=60)
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument("--strategy", choices=("beam", "repair"), default="beam")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--base-precision", choices=("mxfp8", "nvfp4"))
    baseline.add_argument(
        "--base-policy",
        type=Path,
        help="JSON {default, overrides} module policy; preserves existing BF16 protections",
    )
    parser.add_argument(
        "--min-low-precision-fraction",
        type=float,
        default=0.0,
        help="Minimum fraction of configurable modules in MXFP8/NVFP4; repair only",
    )
    parser.add_argument(
        "--require-mixed",
        action="store_true",
        help="Require both BF16 and low-precision modules; repair only",
    )
    parser.add_argument(
        "--min-loss-improvement",
        type=float,
        default=0.0001,
        help="Minimum paired restoration benefit relative to BF16 mean loss",
    )
    parser.add_argument(
        "--quality-criterion", choices=("mean", "block_upper_bound"), default="block_upper_bound"
    )
    parser.add_argument("--repair-steps", type=int, default=8)
    parser.add_argument("--short-steps", type=int, default=16)
    parser.add_argument("--final-steps", type=int, default=64)
    parser.add_argument(
        "--quality-tolerance",
        type=float,
        default=0.005,
        help="Allowed relative validation-loss increase versus BF16 (default 0.5%%)",
    )
    parser.add_argument(
        "--metric", help="Validation loss key; automatically selects lm loss if present"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        base_policy = None
        if args.base_policy:
            value = json.loads(args.base_policy.read_text(encoding="utf-8-sig"))
            if set(value) - {"default", "overrides"} or "default" not in value:
                raise ValueError(
                    "--base-policy accepts module assignments {default, overrides}, not an arbitrary TE recipe"
                )
            base_policy = Policy.from_dict(value)
        elif args.base_precision:
            base_policy = Policy(args.base_precision)
        options = SearchOptions(
            tolerance=args.quality_tolerance,
            max_trials=args.max_trials,
            metric=args.metric,
            short_steps=args.short_steps,
            final_steps=args.final_steps,
            strategy=args.strategy,
            base_policy=base_policy,
            quality_criterion=args.quality_criterion,
            min_low_precision_fraction=args.min_low_precision_fraction,
            require_mixed=args.require_mixed,
            min_loss_improvement=args.min_loss_improvement,
            repair_steps=args.repair_steps,
        )
        runner = CommandRunner(command, args.output, args.budget_minutes * 60, args.resume)
        search = Search(runner, args.output.resolve(), options)
        report = search.run()
    except (ValueError, RuntimeError) as error:
        print(f"Precision search: {error}", file=sys.stderr)
        if "search" in locals():
            from .model import write_json

            report = search.report("failed")
            report["reason"] = str(error)
            write_json(args.output / "report.json", report)
        return 2
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
    return 0 if report["status"] == "verified" else 3


if __name__ == "__main__":
    raise SystemExit(main())
