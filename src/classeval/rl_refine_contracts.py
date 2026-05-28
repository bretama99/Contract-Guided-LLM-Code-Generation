from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable


def add_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def add_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def add_range(cmd: list[str], args: argparse.Namespace) -> None:
    add_optional(cmd, "--start", args.start)
    add_optional(cmd, "--count", args.count)


def run(module: str, args: Iterable[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def base_args(args: argparse.Namespace) -> list[str]:
    cmd = ["--provider", args.provider, "--model", args.model]
    add_range(cmd, args)
    add_bool(cmd, "--overwrite", args.overwrite)
    return cmd


def main() -> None:
    args = parse_args()

    if not args.skip_v1:
        run(
            "src.classeval.generate_contracts",
            [
                *base_args(args),
                "--temperature", str(args.contract_temperature),
                "--max-tokens", str(args.contract_max_tokens),
            ],
            args.dry_run,
        )

        run(
            "src.classeval.generate_from_contracts",
            [
                *base_args(args),
                "--contract-provider", args.provider,
                "--contract-model", args.model,
                "--temperature", str(args.code_temperature),
                "--max-tokens", str(args.code_max_tokens),
            ],
            args.dry_run,
        )

        run(
            "src.classeval.evaluate_contract_guided",
            [
                *base_args(args),
                "--timeout", str(args.timeout),
            ],
            args.dry_run,
        )

    refine_cmd = [
        *base_args(args),
        "--base-provider", args.provider,
        "--base-model", args.model,
        "--temperature", str(args.refine_temperature),
        "--max-tokens", str(args.contract_max_tokens),
    ]
    add_bool(refine_cmd, "--revise-passed", args.revise_passed)
    add_bool(refine_cmd, "--debug", args.debug)

    run("src.classeval.rl_generate_optimized_contracts", refine_cmd, args.dry_run)

    run(
        "src.classeval.generate_from_optimized_contracts",
        [
            *base_args(args),
            "--contract-provider", args.provider,
            "--contract-model", args.model,
            "--temperature", str(args.code_temperature),
            "--max-tokens", str(args.code_max_tokens),
        ],
        args.dry_run,
    )

    run(
        "src.classeval.evaluate_contract_guided_optimized",
        [
            *base_args(args),
            "--timeout", str(args.timeout),
        ],
        args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ClassEval v1 contract pipeline, then feedback-refined v2 contract pipeline."
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)

    parser.add_argument("--contract-temperature", type=float, default=0.0)
    parser.add_argument("--refine-temperature", type=float, default=0.2)
    parser.add_argument("--code-temperature", type=float, default=0.0)

    parser.add_argument("--contract-max-tokens", type=int, default=4096)
    parser.add_argument("--code-max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=15.0)

    parser.add_argument("--skip-v1", action="store_true")
    parser.add_argument("--revise-passed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()