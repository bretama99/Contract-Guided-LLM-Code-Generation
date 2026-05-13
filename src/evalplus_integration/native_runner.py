import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.eval import PASS
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import check_correctness, get_groundtruth


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["task_id"])] = row

    return rows


def load_problems(
    dataset: str,
    mini: bool,
    noextreme: bool,
    version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if dataset == "humaneval":
        problems = get_human_eval_plus(
            mini=mini,
            noextreme=noextreme,
            version=version,
        )
        digest = get_human_eval_plus_hash(
            mini=mini,
            noextreme=noextreme,
            version=version,
        )
        expected = get_groundtruth(problems, digest, [])
        return problems, expected

    if dataset == "mbpp":
        problems = get_mbpp_plus(
            mini=mini,
            noextreme=noextreme,
            version=version,
        )
        digest = get_mbpp_plus_hash(
            mini=mini,
            noextreme=noextreme,
            version=version,
        )
        expected = get_groundtruth(
            problems,
            digest,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )
        return problems, expected

    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


def is_pass(status: Any) -> bool:
    return status == PASS or str(status).lower() == "pass"


def classify_error(status: Any, stage: str) -> tuple[str, str, str]:
    """
    Returns:
        error_type: machine-readable error category
        error_reason: human-readable explanation
        error_stage: generation/base/plus/evaluation
    """
    text = str(status or "").lower()

    if "timeout" in text or "timed out" in text:
        return (
            "timeout",
            f"{stage.title()} evaluation timed out.",
            stage,
        )

    if "syntax" in text:
        return (
            "syntax_error",
            f"{stage.title()} evaluation failed because generated code has a syntax error.",
            stage,
        )

    if "name" in text:
        return (
            "name_error",
            f"{stage.title()} evaluation failed because a required name or symbol was missing.",
            stage,
        )

    if "import" in text or "module" in text:
        return (
            "import_error",
            f"{stage.title()} evaluation failed because of an import/module error.",
            stage,
        )

    if "exception" in text or "error" in text:
        return (
            "runtime_error",
            f"{stage.title()} evaluation raised a runtime error: {status}",
            stage,
        )

    if stage == "base":
        return (
            "base_test_failure",
            f"Base tests failed with status: {status}",
            "base",
        )

    if stage == "plus":
        return (
            "plus_test_failure",
            f"Extra EvalPlus tests failed with status: {status}",
            "plus",
        )

    return (
        "evaluation_failure",
        f"{stage.title()} evaluation failed with status: {status}",
        stage,
    )


def make_detail(
    *,
    task_id: str,
    passed: bool,
    base_passed: bool,
    plus_passed: bool,
    error_type: str | None,
    error_reason: str | None,
    error_stage: str | None,
    base_status: Any,
    plus_status: Any,
) -> dict[str, Any]:
    """
    Keeps old fields failure_type/error for compatibility and adds clearer
    error_type/error_reason/error_stage fields for analysis.
    """
    return {
        "task_id": task_id,
        "passed": passed,
        "base_passed": base_passed,
        "plus_passed": plus_passed,
        "failure_type": error_type,
        "error": error_reason,
        "error_type": error_type,
        "error_reason": error_reason,
        "error_stage": error_stage,
        "base_status": None if base_status is None else str(base_status),
        "plus_status": None if plus_status is None else str(plus_status),
    }

def compact_result(raw: dict[str, Any], base_only: bool) -> dict[str, Any]:
    task_id = str(raw["task_id"])

    raw_debug = {
        "base": repr(raw.get("base")),
        "plus": repr(raw.get("plus")),
    }

    base_status = raw["base"][0]
    plus_status = None if base_only else raw["plus"][0]

    base_passed = is_pass(base_status)
    plus_passed = True if base_only else is_pass(plus_status)
    passed = base_passed and plus_passed

    if passed:
        result = make_detail(
            task_id=task_id,
            passed=True,
            base_passed=True,
            plus_passed=True,
            error_type=None,
            error_reason=None,
            error_stage=None,
            base_status=base_status,
            plus_status=plus_status,
        )
        result["raw_evalplus_debug"] = raw_debug
        return result

    if not base_passed:
        error_type, error_reason, error_stage = classify_error(base_status, "base")
        result = make_detail(
            task_id=task_id,
            passed=False,
            base_passed=False,
            plus_passed=False,
            error_type=error_type,
            error_reason=error_reason,
            error_stage=error_stage,
            base_status=base_status,
            plus_status=plus_status,
        )
        result["raw_evalplus_debug"] = raw_debug
        return result

    error_type, error_reason, error_stage = classify_error(plus_status, "plus")
    result = make_detail(
        task_id=task_id,
        passed=False,
        base_passed=True,
        plus_passed=False,
        error_type=error_type,
        error_reason=error_reason,
        error_stage=error_stage,
        base_status=base_status,
        plus_status=plus_status,
    )
    result["raw_evalplus_debug"] = raw_debug
    return result

def eval_one(
    dataset: str,
    task_id: str,
    problem: dict[str, Any],
    solution: str,
    expected: dict[str, Any],
    base_only: bool,
    test_details: bool,
    min_time_limit: float,
    gt_time_limit_factor: float,
) -> dict[str, Any]:
    try:
        raw = check_correctness(
            dataset,
            0,
            problem,
            solution,
            expected,
            base_only,
            not test_details,
            task_id,
            min_time_limit,
            gt_time_limit_factor,
        )
        return compact_result(raw, base_only)

    except Exception as exc:
        return make_detail(
            task_id=task_id,
            passed=False,
            base_passed=False,
            plus_passed=False,
            error_type="evaluation_error",
            error_reason=f"EvalPlus evaluator crashed: {exc}",
            error_stage="evaluation",
            base_status=None,
            plus_status=None,
        )


def apply_generation_status(
    result: dict[str, Any],
    skipped: dict[str, str],
) -> dict[str, Any]:
    task_id = result["task_id"]

    if task_id in skipped and result.get("passed") is not True:
        reason = skipped[task_id]
        message = (
            "Generation was missing or failed; placeholder failing solution "
            "was evaluated so the task counts as failure."
        )

        result.update(
            {
                "generation_status": "filled_failure",
                "original_generation_failure_type": reason,
                "failure_type": reason,
                "error": message,
                "error_type": reason,
                "error_reason": message,
                "error_stage": "generation",
            }
        )
    else:
        result["generation_status"] = "generated"

    return result


def run_task_level_evalplus(
    *,
    evalplus_dataset: str,
    samples_path: Path,
    export_summary: dict[str, Any],
    parallel: int,
    base_only: bool,
    test_details: bool,
    mini: bool,
    noextreme: bool,
    version: str,
    min_time_limit: float,
    gt_time_limit_factor: float,
) -> list[dict[str, Any]]:
    problems, expected = load_problems(
        evalplus_dataset,
        mini,
        noextreme,
        version,
    )
    samples = read_jsonl(samples_path)

    missing = [task_id for task_id in problems if task_id not in samples]
    if missing:
        raise ValueError(f"Complete sample file is missing tasks: {missing[:10]}")

    skipped = {
        str(item["task_id"]): str(item["reason"])
        for item in export_summary.get("skipped", [])
    }

    task_ids = list(problems)
    index_of = {task_id: index for index, task_id in enumerate(task_ids)}
    results: list[dict[str, Any]] = []

    print("\n[Task-level EvalPlus execution]")

    with ProcessPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {
            pool.submit(
                eval_one,
                evalplus_dataset,
                task_id,
                problems[task_id],
                samples[task_id]["solution"],
                expected[task_id],
                base_only,
                test_details,
                min_time_limit,
                gt_time_limit_factor,
            ): task_id
            for task_id in task_ids
        }

        for future in as_completed(futures):
            result = apply_generation_status(future.result(), skipped)
            result["index"] = index_of[result["task_id"]]
            results.append(result)

            if result["passed"]:
                print(f"[{result['index']}] PASS {result['task_id']}")
            else:
                print(
                    f"[{result['index']}] "
                    f"FAIL/{result['error_type']} "
                    f"{result['task_id']} - {result['error_reason']}"
                )

    return sorted(results, key=lambda row: row["index"])


def summarize_evalplus_details(
    *,
    benchmark: str,
    dataset: str,
    evalplus_dataset: str,
    method: str,
    provider: str,
    model: str,
    details: list[dict[str, Any]],
    export_summary: dict[str, Any],
    base_only: bool,
) -> dict[str, Any]:
    total = len(details)
    base_passed = sum(row["base_passed"] for row in details)
    plus_passed = sum(row["passed"] for row in details)

    failures = Counter(
        row.get("error_type") or row.get("failure_type") or "unknown_failure"
        for row in details
        if not row["passed"]
    )

    base_score = base_passed / total if total else 0.0
    plus_score = plus_passed / total if total else 0.0

    return {
        "benchmark": benchmark,
        "dataset": dataset,
        "evalplus_dataset": evalplus_dataset,
        "method": method,
        "provider": provider,
        "model": model,
        "total_tasks": total,
        "successful_generation_count": export_summary["successful_sample_count"],
        "filled_failure_count": export_summary["filled_failure_count"],
        "base_passed": base_passed,
        "plus_passed": plus_passed,
        "failed": total - plus_passed,
        "base_pass@1": base_score,
        "base_pass@1_percent": round(base_score * 100, 2),
        "plus_pass@1": None if base_only else plus_score,
        "plus_pass@1_percent": None if base_only else round(plus_score * 100, 2),
        "failure_counts": dict(failures),
        "base_only": base_only,
    }