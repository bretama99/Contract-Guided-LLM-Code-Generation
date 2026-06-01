from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from src.common.config import DATASETS
from src.common.io_utils import load_json, load_json_list, save_json
from src.common.llm_clients import default_model
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_results_folder
from src.common.task_utils import select_tasks, task_identifier
from src.classeval.execution import short
from src.livecodebench.evaluate_vanilla import cases, norm, run_code

STAGE = "stage_2c_livecodebench_contract_guided_evaluation"
METHOD = "livecodebench_contract_guided_generation"

def load_contract(generation: dict[str, Any]) -> Any:
    path = generation.get("contract_path")

    if not path:
        return None

    try:
        record = load_json(Path(path))
    except Exception:
        return None

    return record.get("contract", record) if isinstance(record, dict) else None


def fail(task: dict[str, Any], generation: dict[str, Any] | None, kind: str, error: str) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "index": None,
        "entry_point": None,
        "passed": False,
        "failure_type": kind,
        "failure_explanation": error,
        "task": {
            "id": task_identifier(task),
            "benchmark": "livecodebench",
            "task_type": task.get("task_type"),
            "prompt": task.get("prompt"),
            "input_output": task.get("input_output"),
            "metadata": task.get("metadata"),
        },
        "generation": generation or {
            "status": kind,
            "error": error,
            "generated_code": None,
            "contract": None,
        },
        "evaluation": {
            "status": "failed",
            "failure_type": kind,
            "error": error,
        },
    }


def evaluate_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task_id = task_identifier(task)
    gen_file = raw_contract_code_path(args.dataset, task_id, args.provider, args.model)

    if not gen_file.exists():
        return fail(task, None, "missing_generation", f"Missing generated file: {gen_file}")

    generation = load_json(gen_file)

    if not isinstance(generation, dict):
        return fail(task, None, "invalid_generation_file", "Generation file is not a JSON object.")

    generation["contract"] = load_contract(generation)

    if generation.get("status") != "success":
        return fail(task, generation, "generation_failed", str(generation.get("error")))

    code = str(generation.get("generated_code") or "")

    if not code.strip():
        return fail(task, generation, "empty_generated_code", "Generated code is empty or missing.")

    test_cases = cases(task)

    if not test_cases:
        return fail(task, generation, "missing_test_cases", "No public stdin/stdout test cases found.")

    for case_index, (stdin, expected) in enumerate(test_cases):
        result = run_code(code, stdin, args.timeout)

        actual = norm(result.get("stdout") or "")
        expected_norm = norm(expected)

        if result.get("ok") is not True:
            failure_type = result.get("failure_type") or "runtime_error"

            return {
                "task_id": task_id,
                "index": None,
                "entry_point": None,
                "passed": False,
                "failure_type": failure_type,
                "failure_explanation": short(result.get("error"), 500),
                "task": {
                    "id": task_id,
                    "benchmark": "livecodebench",
                    "task_type": task.get("task_type"),
                    "prompt": task.get("prompt"),
                    "input_output": task.get("input_output"),
                    "metadata": task.get("metadata"),
                },
                "generation": generation,
                "evaluation": {
                    "status": "failed",
                    "failure_type": failure_type,
                    "returncode": result.get("returncode"),
                    "stdout": short(result.get("stdout"), 3000),
                    "stderr": short(result.get("stderr"), 6000),
                    "error": result.get("error"),
                    "case_index": case_index,
                    "input": stdin,
                    "expected": expected_norm,
                    "actual": actual,
                },
            }

        if actual != expected_norm:
            return {
                "task_id": task_id,
                "index": None,
                "entry_point": None,
                "passed": False,
                "failure_type": "wrong_answer",
                "failure_explanation": f"Expected {expected_norm!r}, got {actual!r}",
                "task": {
                    "id": task_id,
                    "benchmark": "livecodebench",
                    "task_type": task.get("task_type"),
                    "prompt": task.get("prompt"),
                    "input_output": task.get("input_output"),
                    "metadata": task.get("metadata"),
                },
                "generation": generation,
                "evaluation": {
                    "status": "failed",
                    "failure_type": "wrong_answer",
                    "case_index": case_index,
                    "input": stdin,
                    "expected": expected_norm,
                    "actual": actual,
                    "stdout": short(result.get("stdout"), 3000),
                    "stderr": short(result.get("stderr"), 6000),
                },
            }

    return {
        "task_id": task_id,
        "index": None,
        "entry_point": None,
        "passed": True,
        "failure_type": None,
        "failure_explanation": "Passed all public LiveCodeBench stdin/stdout test cases.",
        "task": {
            "id": task_id,
            "benchmark": "livecodebench",
            "task_type": task.get("task_type"),
            "prompt": task.get("prompt"),
            "input_output": task.get("input_output"),
            "metadata": task.get("metadata"),
        },
        "generation": generation,
        "evaluation": {
            "status": "passed",
            "failure_type": None,
            "num_cases": len(test_cases),
        },
    }


def summarize(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(row.get("passed") is True for row in results)

    failures = Counter(
        str(row.get("failure_type") or "unknown")
        for row in results
        if row.get("passed") is not True
    )

    missing_tests = failures.get("missing_test_cases", 0)
    testable_total = total - missing_tests

    pass_at_1 = passed / total if total else 0.0
    pass_at_1_testable = passed / testable_total if testable_total else 0.0

    return {
        "benchmark": "LiveCodeBench",
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model,
        "method": METHOD,
        "stage": STAGE,
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass@1": pass_at_1,
        "pass@1_percent": round(pass_at_1 * 100, 2),
        "testable_total": testable_total,
        "pass@1_testable": pass_at_1_testable,
        "pass@1_testable_percent": round(pass_at_1_testable * 100, 2),
        "timeout_seconds": args.timeout,
        "failure_counts": dict(failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiveCodeBench Stage 2C: evaluate contract-guided generations"
    )

    parser.add_argument("--dataset", default="livecodebench")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = "livecodebench"
    args.model = args.model or default_model(args.provider)

    info = DATASETS[args.dataset]
    tasks = select_tasks(
        load_json_list(info["path"]),
        start=args.start,
        count=args.count,
    )

    results = []

    for index, task in enumerate(tasks, start=args.start):
        result = evaluate_one(task, args)
        result["index"] = index
        results.append(result)

        status = "PASS" if result.get("passed") is True else f"FAIL/{result.get('failure_type', 'unknown')}"
        print(f"[{index}] {status} {result['task_id']}")

    summary = summarize(args, results)
    out_dir = raw_contract_results_folder(args.dataset, args.provider, args.model)

    save_json(out_dir / "livecodebench_summary.json", summary)
    save_json(out_dir / "livecodebench_details.json", results)

    print("\nLiveCodeBench contract-guided evaluation finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()