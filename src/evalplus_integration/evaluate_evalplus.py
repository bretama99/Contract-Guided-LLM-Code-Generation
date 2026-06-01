#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.config import DATASETS
from src.common.io_utils import save_json
from src.evalplus_integration.native_runner import (
    run_task_level_evalplus,
    summarize_evalplus_details,
)
from src.evalplus_integration.paths import SUPPORTED_METHODS, results_folder
from src.evalplus_integration.samples import export_evalplus_samples


def evalplus_dataset_keys() -> list[str]:
    return sorted(key for key, info in DATASETS.items() if "evalplus_dataset" in info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EvalPlus HumanEval+ or MBPP+ generations")
    parser.add_argument("--dataset", choices=evalplus_dataset_keys(), default="evalplus")
    parser.add_argument("--method", choices=sorted(SUPPORTED_METHODS), default="vanilla")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--noextreme", action="store_true")
    parser.add_argument("--version", default="default")
    parser.add_argument("--min-time-limit", type=float, default=1.0)
    parser.add_argument("--gt-time-limit-factor", type=float, default=4.0)
    return parser.parse_args()


def print_export(summary: dict[str, Any], successful: Path, complete: Path, export_path: Path) -> None:
    print(f"Successful-only samples: {successful}")
    print(f"Complete samples:        {complete}")
    print(f"Export summary:          {export_path}")
    print(f"Successful samples: {summary['successful_sample_count']} / {summary['task_count']}")
    print(f"Complete samples:   {summary['complete_sample_count']} / {summary['task_count']}")

    skipped = summary.get("skipped", [])
    if skipped:
        print(f"Filled {len(skipped)} missing/failed generations as explicit failures.")
    else:
        print("No missing or failed generations.")


def save_results(
    out_dir: Path,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    export_summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "details.json", details)
    save_json(out_dir / "export_summary.json", export_summary)

    print(f"Summary:        {out_dir / 'summary.json'}")
    print(f"Details:        {out_dir / 'details.json'}")
    print(f"Export summary: {out_dir / 'export_summary.json'}")


def main() -> None:
    args = parse_args()

    if args.count is not None and not args.export_only:
        raise SystemExit("Use --count only with --export-only. Final EvalPlus evaluation needs the full benchmark.")

    dataset_info = DATASETS[args.dataset]
    evalplus_dataset = dataset_info["evalplus_dataset"]

    print("\n[1/4] Exporting EvalPlus samples")
    successful, complete, export_path, export_summary = export_evalplus_samples(
        dataset=args.dataset,
        method=args.method,
        provider=args.provider,
        model=args.model,
        start=args.start,
        count=args.count,
    )
    print_export(export_summary, successful, complete, export_path)

    if args.export_only:
        print("\nExport-only mode enabled.")
        return

    print("\n[2/4] Running task-level EvalPlus evaluation")
    details = run_task_level_evalplus(
        evalplus_dataset=evalplus_dataset,
        samples_path=complete,
        export_summary=export_summary,
        parallel=args.parallel,
        base_only=args.base_only,
        test_details=args.test_details,
        mini=args.mini,
        noextreme=args.noextreme,
        version=args.version,
        min_time_limit=args.min_time_limit,
        gt_time_limit_factor=args.gt_time_limit_factor,
    )

    print("\n[3/4] Summarizing results")
    summary = summarize_evalplus_details(
        benchmark=dataset_info.get("label", args.dataset),
        dataset=args.dataset,
        evalplus_dataset=evalplus_dataset,
        method=args.method,
        provider=args.provider,
        model=args.model,
        details=details,
        export_summary=export_summary,
        base_only=args.base_only,
    )

    out_dir = results_folder(args.method, args.dataset, args.provider, args.model)
    summary["artifacts"] = {
        "successful_samples_path": str(successful),
        "complete_samples_path": str(complete),
        "export_summary_path": str(export_path),
        "summary_path": str(out_dir / "summary.json"),
        "details_path": str(out_dir / "details.json"),
    }

    print("\n[4/4] Saving artifacts")
    save_results(out_dir, summary, details, export_summary)

    print("\nEvalPlus evaluation finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()