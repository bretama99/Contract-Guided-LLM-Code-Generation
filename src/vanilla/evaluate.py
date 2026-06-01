#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.common.benchmarks import evaluate_candidate, get_benchmark, require_native_evaluator
from src.common.config import DATASETS, OUTPUT_ROOT, RESULTS_ROOT
from src.common.execution import normalize_failure_type
from src.common.io_utils import load_json, load_json_list, safe_name, save_json
from src.common.llm_clients import default_model
from src.common.task_utils import select_tasks, task_entry_point, task_identifier, task_prompt

STAGE = "vanilla_evaluation"
METHOD = "vanilla"

GEN_DIR = OUTPUT_ROOT / "vanilla"
EVAL_DIR = OUTPUT_ROOT / "evaluation" / "vanilla"
RESULT_DIR = RESULTS_ROOT / "vanilla"


def generation_folder(dataset: str, provider: str, model: str) -> Path:
    return GEN_DIR / safe_name(provider) / safe_name(model) / safe_name(dataset)


def generation_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return generation_folder(dataset, provider, model) / f"{safe_name(task_id)}_vanilla.json"


def evaluation_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return EVAL_DIR / safe_name(provider) / safe_name(model) / safe_name(dataset) / f"{safe_name(task_id)}_eval.json"


def short(value: Any, limit: int = 3000) -> str:
    value = "" if value is None else str(value).strip()
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def task_info(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task_identifier(task),
        "entry_point": task_entry_point(task),
        "prompt": task_prompt(task),
    }


def generation_info(generation: dict[str, Any] | None) -> dict[str, Any]:
    generation = generation or {}
    return {
        "status": generation.get("status"),
        "error": generation.get("error"),
        "code": generation.get("generated_code"),
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


def explanation(passed: bool, failure_type: str | None, generation: dict[str, Any], evaluation: dict[str, Any]) -> str:
    if passed:
        return "Passed all evaluated benchmark tests."

    if generation.get("status") != "success":
        return f"Generation failed before evaluation: {generation.get('error')}."

    if evaluation.get("error"):
        return f"Evaluation failed with {failure_type}: {short(evaluation.get('error'), 500)}"

    return f"Failed with failure type: {failure_type or 'other'}."


def make_result(
    *,
    task: dict[str, Any],
    generation: dict[str, Any] | None,
    passed: bool,
    failure_type: str | None,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    gen = generation_info(generation)

    return {
        "task_id": task_identifier(task),
        "index": None,
        "entry_point": task_entry_point(task),
        "stage": STAGE,
        "method": METHOD,
        "passed": passed,
        "failure_type": failure_type,
        "failure_explanation": explanation(passed, failure_type, gen, evaluation),
        "task": task_info(task),
        "generation": gen,
        "evaluation": evaluation,
    }


def fail(task: dict[str, Any], generation: dict[str, Any] | None, failure_type: str, error: str) -> dict[str, Any]:
    normalized = normalize_failure_type(failure_type)
    return make_result(
        task=task,
        generation=generation,
        passed=False,
        failure_type=normalized,
        evaluation=evaluation_info(None, normalized, error),
    )


def evaluate_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tid = task_identifier(task)
    gen_file = generation_path(args.dataset, tid, args.provider, args.model)

    if not gen_file.exists():
        return fail(task, None, "other", f"Missing generated file: {gen_file}")

    generation = load_json(gen_file)
    if not isinstance(generation, dict):
        return fail(task, None, "other", f"Generation file is not a JSON object: {gen_file}")

    if generation.get("status") != "success":
        return fail(task, generation, "other", str(generation.get("error") or "Generation failed."))

    code = generation.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        return fail(task, generation, "other", "Generated code is empty or missing.")

    benchmark = get_benchmark(args.dataset)
    result = evaluate_candidate(
        benchmark=benchmark,
        task=task,
        code=code,
        timeout=args.timeout,
    )

    passed = result.get("passed") is True
    failure_type = None if passed else normalize_failure_type(result.get("failure_type"))
    evaluation = evaluation_info(result, failure_type, result.get("error"))

    return make_result(
        task=task,
        generation=generation,
        passed=passed,
        failure_type=failure_type,
        evaluation=evaluation,
    )


def summarize(args: argparse.Namespace, benchmark: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(row.get("passed") is True for row in results)

    failures = Counter(
        row.get("failure_type") or "passed"
        for row in results
        if row.get("passed") is not True
    )

    score = passed / total if total else 0.0

    return {
        "benchmark": benchmark,
        "dataset": args.dataset,
        "provider": args.provider,
        "model_name": args.model,
        "method": METHOD,
        "stage": STAGE,
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass@1": score,
        "pass@1_percent": round(score * 100, 2),
        "timeout_seconds": args.timeout,
        "failure_counts": dict(failures),
    }


def process_one(index: int, task: dict[str, Any], args: argparse.Namespace) -> tuple[int, dict[str, Any], str]:
    tid = task_identifier(task)
    out = evaluation_path(args.dataset, tid, args.provider, args.model)

    if out.exists() and not args.overwrite:
        result = load_json(out)
    else:
        result = evaluate_one(task, args)
        result["index"] = index
        save_json(out, result)

    result["index"] = index
    status = "PASS" if result.get("passed") else f"FAIL/{result.get('failure_type')}"
    return index, result, f"[{index}] {status}: {tid}"


def run(args: argparse.Namespace) -> None:
    args.model = args.model or default_model(args.provider)

    benchmark = get_benchmark(args.dataset)
    require_native_evaluator(benchmark, method=METHOD)

    tasks = select_tasks(load_json_list(DATASETS[args.dataset]["path"]), start=args.start, count=args.count)

    started = time.perf_counter()
    indexed: list[tuple[int, dict[str, Any]]] = []

    if args.workers <= 1:
        for index, task in enumerate(tasks, start=args.start):
            item_index, result, message = process_one(index, task, args)
            print(message)
            indexed.append((item_index, result))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one, index, task, args): index
                for index, task in enumerate(tasks, start=args.start)
            }

            for future in as_completed(futures):
                item_index, result, message = future.result()
                print(message)
                indexed.append((item_index, result))

    indexed.sort(key=lambda row: row[0])
    results = [row[1] for row in indexed]

    summary = summarize(args, benchmark.label, results)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 4)

    name = f"stage1_{args.dataset}_{safe_name(args.provider)}_{safe_name(args.model)}"
    save_json(RESULT_DIR / f"{name}_summary.json", summary)
    save_json(RESULT_DIR / f"{name}_details.json", results)

    print("\nStage 1 vanilla evaluation finished")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vanilla generated code")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())