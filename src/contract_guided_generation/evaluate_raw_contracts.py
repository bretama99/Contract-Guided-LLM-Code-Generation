import argparse
import json
from collections import Counter
from typing import Any

from src.common.config import DATASETS
from src.common.execution import evaluate_humaneval_candidate, evaluate_bigcodebench_candidate
from src.common.io_utils import load_json, load_json_list, save_json
from src.common.llm_clients import default_model
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_results_folder
from src.common.task_utils import select_tasks, task_identifier

STAGE = "stage_2c_raw_contract_guided_evaluation"
METHOD = "raw_contract_guided_generation"


def fail(task: dict[str, Any], gen_file, kind: str, error: str) -> dict[str, Any]:
    return {
        "task_id": str(task["task_id"]),
        "entry_point": task.get("entry_point"),
        "generation_file": str(gen_file),
        "stage": STAGE,
        "passed": False,
        "failure_type": kind,
        "error": error,
        "stdout": "",
        "stderr": "",
    }


def evaluate_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task_id = task_identifier(task)
    gen_file = raw_contract_code_path(args.dataset, task_id, args.provider, args.model)

    if not gen_file.exists():
        return fail(task, gen_file, "missing_generation", f"Missing generated file: {gen_file}")

    generation = load_json(gen_file)
    if not isinstance(generation, dict):
        return fail(task, gen_file, "invalid_generation_file", "Generation file is not a JSON object.")

    if generation.get("status") != "success":
        return fail(task, gen_file, "generation_failed", str(generation.get("error")))

    code = generation.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        return fail(task, gen_file, "empty_generated_code", "Generated code is empty or missing.")

    return {
        "task_id": task_id,
        "entry_point": task.get("entry_point"),
        "generation_file": str(gen_file),
        "stage": STAGE,
        **evaluate_candidate(args.dataset, task, code, args.timeout),
    }

def evaluate_candidate(
    dataset: str,
    task: dict[str, Any],
    code: str,
    timeout: float,
) -> dict[str, Any]:
    if "evalplus_dataset" in DATASETS[dataset]:
        raise SystemExit(
            "EvalPlus datasets must be evaluated with: "
            "python scripts/evaluate_evalplus.py --method raw_contracts"
        )

    if dataset == "humaneval":
        return evaluate_humaneval_candidate(task, code, timeout)

    if dataset == "bigcodebench":
        return evaluate_bigcodebench_candidate(task, code, timeout)

    raise NotImplementedError(
        f"No raw-contract evaluator implemented for dataset: {dataset}"
    )
    
def summarize(
    benchmark: str,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    passed = sum(row.get("passed") is True for row in results)
    failures = Counter(
        str(row.get("failure_type", "unknown"))
        for row in results
        if row.get("passed") is not True
    )
    pass_at_1 = passed / total if total else 0.0

    return {
        "benchmark": benchmark,
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model,
        "method": METHOD,
        "stage": STAGE,
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "missing": failures.get("missing_generation", 0),
        "pass@1": pass_at_1,
        "pass@1_percent": round(pass_at_1 * 100, 2),
        "timeout_seconds": args.timeout,
        "failure_counts": dict(failures),
    }


def evaluate(args: argparse.Namespace) -> None:
    if "evalplus_dataset" in DATASETS[args.dataset]:
        raise SystemExit(
            "Use scripts/evaluate_evalplus.py --method raw_contracts for EvalPlus datasets."
        )

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
        results.append(result)

        status = (
            "PASS"
            if result.get("passed") is True
            else f"FAIL/{result.get('failure_type', 'unknown')}"
        )
        print(f"[{index}] {status} {result['task_id']}")

    summary = summarize(info["label"], args, results)
    out_dir = raw_contract_results_folder(args.dataset, args.provider, args.model)

    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "details.json", results)

    print("\nStage 2C raw contract-guided evaluation finished")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2C: evaluate raw contract-guided generations"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="humaneval")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())