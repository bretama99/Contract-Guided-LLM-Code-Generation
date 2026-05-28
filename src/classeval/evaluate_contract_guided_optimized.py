from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.classeval.core import JsonDict, task_id
from src.classeval.execution import evaluate_generation, failure_record, summarize_results
from src.classeval.generate_contracts import TASK_FILE
from src.classeval.generate_from_optimized_contracts import generation_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model
from src.common.task_utils import select_tasks

STAGE = "2R_feedback_optimized_contract_guided_evaluation"

EVAL_DIR = OUTPUT_ROOT / "classeval" / "evaluation" / "rl_optimized_contract_guided"
RESULT_DIR = RESULTS_ROOT / "classeval" / "rl_optimized_contract_guided"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contract_guided_eval.log"


def evaluation_path(task: JsonDict, provider: str, model: str) -> Path:
    return EVAL_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def compact_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = value.strip()
    return value[:limit] + ("...[truncated]" if len(value) > limit else "")


def important_lines(stderr: str, limit: int = 40) -> list[str]:
    markers = (
        "FAIL:",
        "ERROR:",
        "AssertionError",
        "TypeError",
        "ValueError",
        "AttributeError",
        "KeyError",
        "IndexError",
        "NameError",
        "ImportError",
        "Timeout",
        "expected",
        "got",
        "not found",
        "NoneType",
        "invalid",
    )

    lines: list[str] = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped and any(marker in stripped for marker in markers):
            lines.append(stripped)
        if len(lines) >= limit:
            break

    return lines


def feedback_for_next_contract(result: JsonDict) -> JsonDict:
    evaluation = result.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        evaluation = {}

    metrics = evaluation.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    stderr = compact_text(evaluation.get("stderr") or result.get("error"), 8000)
    selected_lines = important_lines(stderr)

    feedback: JsonDict = {
        "passed": result.get("passed") is True,
        "failure_type": result.get("failure_type"),
        "tests_total": metrics.get("total"),
        "tests_passed": metrics.get("passed"),
        "tests_failed": metrics.get("failed"),
        "failures": metrics.get("failures"),
        "errors": metrics.get("errors"),
        "skipped": metrics.get("skipped"),
        "failure_summary": compact_text("\n".join(selected_lines), 3500),
    }

    if result.get("error"):
        feedback["evaluation_error"] = compact_text(result.get("error"), 1500)

    return {k: v for k, v in feedback.items() if v not in (None, "", [], {})}


def generation_status(record: JsonDict) -> JsonDict:
    return {
        "generation_status": record.get("status"),
        "generation_error": compact_text(record.get("error"), 1500),
        "copied_from_v1": record.get("copied_from_v1") is True,
    }


def evaluate_one(task: JsonDict, provider: str, model: str, timeout: float, include_code: bool) -> JsonDict:
    gen_path = generation_path(task, provider, model)

    if not gen_path.exists():
        result = failure_record(
            task,
            provider=provider,
            model=model,
            stage=STAGE,
            failure_type="missing_generation",
            error=f"Missing generation file: {gen_path}",
            generation_path=gen_path,
        )
        result["feedback_for_next_contract"] = feedback_for_next_contract(result)
        return result

    generation = load_json(gen_path)

    result = evaluate_generation(
        task,
        generation,
        gen_path,
        provider=provider,
        model=model,
        stage=STAGE,
        timeout=timeout,
        include_code=include_code,
    )

    result["generation"] = {
        "path": str(gen_path),
        **generation_status(generation),
    }
    result["feedback_for_next_contract"] = feedback_for_next_contract(result)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one code solution per optimized contract.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    results: list[JsonDict] = []
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        out_path = evaluation_path(task, args.provider, model)

        if out_path.exists() and not args.overwrite:
            result = load_json(out_path)
        else:
            result = evaluate_one(
                task=task,
                provider=args.provider,
                model=model,
                timeout=args.timeout,
                include_code=args.include_code,
            )
            save_json(out_path, result)

        result["index"] = index
        results.append(result)

        feedback = result.get("feedback_for_next_contract") or {}
        status = "PASS" if result.get("passed") else f"FAIL/{result.get('failure_type')}"
        passed = feedback.get("tests_passed")
        total = feedback.get("tests_total")
        metric_text = f" tests={passed}/{total}" if passed is not None and total is not None else ""

        print(f"[{index}] {status}: {task_id(task)}{metric_text}")

        if args.fail_fast and result.get("passed") is not True:
            break

    summary = summarize_results(
        results,
        provider=args.provider,
        model=model,
        stage=STAGE,
        timeout=args.timeout,
    )
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 4)

    name = f"classeval_feedback_optimized_{safe_name(args.provider)}_{safe_name(model)}"
    save_json(RESULT_DIR / f"{name}_summary.json", summary)
    save_json(RESULT_DIR / f"{name}_details.json", results)

    print("\nClassEval optimized-contract-guided evaluation finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()