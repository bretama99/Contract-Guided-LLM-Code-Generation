#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.common.benchmarks import evaluate_candidate, get_benchmark, require_native_evaluator
from src.common.config import DATASETS
from src.common.execution import normalize_failure_type
from src.common.io_utils import load_json, load_json_list, save_json
from src.common.llm_clients import default_model
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_results_folder
from src.common.task_utils import select_tasks, task_entry_point, task_identifier, task_prompt
from src.rl_method_level.rl_generate_contract import optimized_contract_code_path, optimized_contract_results_folder

CONTRACT_SOURCES = ("raw", "optimized_rl")


def require_source(value: str) -> str:
    if value not in CONTRACT_SOURCES:
        raise ValueError(f"Unsupported contract source: {value}")
    return value


def stage_for(source: str) -> str:
    return "stage_2c_raw_contract_guided_evaluation" if require_source(source) == "raw" else "stage_2f_optimized_rl_contract_guided_evaluation"


def method_for(source: str) -> str:
    return "raw_contract_guided_generation" if require_source(source) == "raw" else "optimized_rl_contract_guided_generation"


def generation_path(source: str, dataset: str, task_id: str, provider: str, model: str):
    return raw_contract_code_path(dataset, task_id, provider, model) if require_source(source) == "raw" else optimized_contract_code_path(dataset, task_id, provider, model)


def result_folder(source: str, dataset: str, provider: str, model: str):
    return raw_contract_results_folder(dataset, provider, model) if require_source(source) == "raw" else optimized_contract_results_folder(dataset, provider, model)


def short(value: Any, limit: int = 3000) -> str:
    text = "" if value is None else str(value).strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def task_info(task: dict[str, Any]) -> dict[str, Any]:
    return {"id": task_identifier(task), "entry_point": task_entry_point(task), "prompt": task_prompt(task)}


def generation_info(generation: dict[str, Any] | None) -> dict[str, Any]:
    generation = generation or {}
    return {
        "status": generation.get("status"),
        "error": generation.get("error"),
        "code": generation.get("generated_code"),
        "contract_path": generation.get("contract_path"),
        "contract_type": generation.get("contract_type"),
        "contract_version": generation.get("contract_version"),
        "contract_optimized": generation.get("contract_optimized"),
        "contract_optimization_status": generation.get("contract_optimization_status"),
        "contract_optimization_changed": generation.get("contract_optimization_changed"),
        "contract_optimized_success": generation.get("contract_optimized_success"),
        "contract_fallback_used": generation.get("contract_fallback_used"),
        "contract_issue": generation.get("contract_issue"),
        "contract_reason": generation.get("contract_reason"),
        "contract_solution": generation.get("contract_solution"),
        "reused_raw_generation": generation.get("reused_raw_generation"),
        "raw_generation_path": generation.get("raw_generation_path"),
    }


def evaluation_info(result: dict[str, Any] | None, failure_type: str | None, error: str | None = None) -> dict[str, Any]:
    result = result or {}
    return {
        "status": "passed" if result.get("passed") is True else "failed",
        "returncode": result.get("returncode"),
        "stdout": short(result.get("stdout"), 3000),
        "stderr": short(result.get("stderr"), 6000),
        "error": short(error or result.get("error"), 6000),
        "failure_type": failure_type,
    }


def explanation(passed: bool, failure_type: str | None, gen: dict[str, Any], ev: dict[str, Any]) -> str:
    if passed:
        return "Passed all evaluated benchmark tests."
    if gen.get("status") != "success":
        return f"Generation failed before evaluation: {gen.get('error')}."
    if ev.get("error"):
        return f"Evaluation failed with {failure_type}: {short(ev.get('error'), 500)}"
    return f"Failed with failure type: {failure_type or 'other'}."


def make_result(*, task: dict[str, Any], source: str, generation: dict[str, Any] | None, passed: bool, failure_type: str | None, evaluation: dict[str, Any]) -> dict[str, Any]:
    gen = generation_info(generation)
    return {
        "task_id": task_identifier(task), "index": None, "entry_point": task_entry_point(task),
        "stage": stage_for(source), "method": method_for(source), "contract_source": require_source(source),
        "passed": passed, "failure_type": failure_type,
        "failure_explanation": explanation(passed, failure_type, gen, evaluation),
        "task": task_info(task), "generation": gen, "evaluation": evaluation,
    }


def fail(task: dict[str, Any], source: str, generation: dict[str, Any] | None, failure_type: str, error: str) -> dict[str, Any]:
    normalized = normalize_failure_type(failure_type)
    return make_result(task=task, source=source, generation=generation, passed=False, failure_type=normalized, evaluation=evaluation_info(None, normalized, error))


def evaluate_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tid = task_identifier(task)
    gen_file = generation_path(args.contract_source, args.dataset, tid, args.provider, args.model)
    if not gen_file.exists():
        return fail(task, args.contract_source, None, "other", f"Missing generated file: {gen_file}")
    generation = load_json(gen_file)
    if not isinstance(generation, dict):
        return fail(task, args.contract_source, None, "other", f"Generation file is not a JSON object: {gen_file}")
    if generation.get("status") != "success":
        return fail(task, args.contract_source, generation, "other", str(generation.get("error") or "Generation failed."))
    code = generation.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        return fail(task, args.contract_source, generation, "other", "Generated code is empty or missing.")
    result = evaluate_candidate(benchmark=get_benchmark(args.dataset), task=task, code=code, timeout=args.timeout)
    passed = result.get("passed") is True
    failure_type = None if passed else normalize_failure_type(result.get("failure_type"))
    evaluation = evaluation_info(result, failure_type, result.get("error"))
    return make_result(task=task, source=args.contract_source, generation=generation, passed=passed, failure_type=failure_type, evaluation=evaluation)


def summarize(args: argparse.Namespace, benchmark: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(row.get("passed") is True for row in results)
    failures = Counter(row.get("failure_type") or "passed" for row in results if row.get("passed") is not True)
    score = passed / total if total else 0.0
    return {
        "benchmark": benchmark, "dataset": args.dataset, "provider": args.provider,
        "model_name": args.model, "contract_source": args.contract_source,
        "method": method_for(args.contract_source), "stage": stage_for(args.contract_source),
        "total_tasks": total, "passed": passed, "failed": total - passed,
        "pass@1": score, "pass@1_percent": round(score * 100, 2),
        "timeout_seconds": args.timeout, "failure_counts": dict(failures),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(args: argparse.Namespace) -> None:
    args.model = args.model or default_model(args.provider)
    benchmark = get_benchmark(args.dataset)
    require_native_evaluator(benchmark, method="raw_contracts" if args.contract_source == "raw" else "optimized_rl")
    tasks = select_tasks(load_json_list(DATASETS[args.dataset]["path"]), start=args.start, count=args.count)
    started = time.perf_counter(); results = []
    for index, task in enumerate(tasks, start=args.start):
        result = evaluate_one(task, args)
        result["index"] = index
        results.append(result)
        status = "PASS" if result.get("passed") else f"FAIL/{result.get('failure_type')}"
        print(f"[{index}] {status} {result['task_id']}")
    summary = summarize(args, benchmark.label, results)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    out_dir = result_folder(args.contract_source, args.dataset, args.provider, args.model)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "details.json", results)
    print(f"\nStage 2 contract-guided evaluation finished [{args.contract_source}]")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2C/2F: evaluate raw or optimized contract-guided generations")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--contract-source", choices=CONTRACT_SOURCES, default="raw")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
