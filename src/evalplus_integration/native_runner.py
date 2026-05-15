import json
import subprocess
import sys
import tempfile
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

from src.common.io_utils import safe_name


def short(value: Any, limit: int = 1500) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def read_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["task_id"])] = row
    return rows


def load_problems(dataset: str, mini: bool, noextreme: bool, version: str):
    if dataset == "humaneval":
        problems = get_human_eval_plus(mini=mini, noextreme=noextreme, version=version)
        digest = get_human_eval_plus_hash(mini=mini, noextreme=noextreme, version=version)
        return problems, get_groundtruth(problems, digest, [])

    if dataset == "mbpp":
        problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
        digest = get_mbpp_plus_hash(mini=mini, noextreme=noextreme, version=version)
        return problems, get_groundtruth(problems, digest, MBPP_OUTPUT_NOT_NONE_TASKS)

    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


def passed_status(status: Any) -> bool:
    return status == PASS or str(status).lower() == "pass"


def passed_value(value: Any) -> bool:
    return value == 1 or value is True or passed_status(value)


def part_status(part: Any) -> Any:
    return part[0] if isinstance(part, (tuple, list)) and part else None


def part_vector(part: Any) -> list[Any]:
    if not isinstance(part, (tuple, list)) or len(part) < 2:
        return []
    return part[1] if isinstance(part[1], list) else []


def first_failed(part: Any) -> int | None:
    for index, value in enumerate(part_vector(part)):
        if not passed_value(value):
            return index
    return None


def get_inputs(problem: dict[str, Any], stage: str) -> list[Any]:
    key = "base_input" if stage == "original" else "plus_input"
    value = problem.get(key)
    return value if isinstance(value, list) else []


def get_expected(expected: Any, stage: str) -> list[Any]:
    source = "base" if stage == "original" else "plus"

    if isinstance(expected, dict):
        for key in (
            source,
            f"{source}_output",
            f"{source}_outputs",
            f"{source}_expected",
            f"{source}_expected_output",
            f"{source}_expected_outputs",
        ):
            value = expected.get(key)
            if isinstance(value, list):
                return value

        outputs = expected.get("outputs")
        if isinstance(outputs, dict) and isinstance(outputs.get(source), list):
            return outputs[source]

    return expected if isinstance(expected, list) else []


def item_at(values: list[Any], index: int | None) -> Any:
    if index is None:
        return None
    return values[index] if 0 <= index < len(values) else None


def generation_file(task_id: str, export_summary: dict[str, Any]) -> Path | None:
    folder = Path(str(export_summary.get("generation_folder", "")))
    method = str(export_summary.get("method", ""))

    if not folder.exists():
        return None

    suffix = "_vanilla.json" if method == "vanilla" else "_raw_contract_guided.json"
    direct = folder / f"{safe_name(task_id)}{suffix}"

    if direct.exists():
        return direct

    matches = sorted(folder.glob(f"*{safe_name(task_id)}{suffix}"))
    return matches[0] if matches else None


def generation_info(task_id: str, solution: str, export_summary: dict[str, Any], skipped: str | None):
    path = generation_file(task_id, export_summary)
    record = read_json(path)
    method = str(export_summary.get("method", ""))

    code = solution
    if record and isinstance(record.get("generated_code"), str):
        code = record["generated_code"]

    info = {
        "status": "generation_failed" if skipped else "generated",
        "error": skipped or (record.get("error") if record else None),
        "code": code,
    }

    if method != "vanilla" and record and record.get("contract_path"):
        info["contract_path"] = record.get("contract_path")

    return info


def task_info(task_id: str, problem: dict[str, Any], method: str) -> dict[str, Any]:
    prompt = (
        problem.get("prompt")
        or problem.get("instruct_prompt")
        or problem.get("complete_prompt")
    )

    info = {
        "id": task_id,
        "entry_point": problem.get("entry_point"),
        "prompt": prompt,
    }

    if method != "vanilla" and problem.get("contract"):
        info["contract"] = problem.get("contract")

    return info


def run_actual(solution: str, entry_point: str | None, case_input: Any, timeout: float = 5.0):
    if not entry_point:
        return None, "missing entry point"

    program = f"""
{solution}

import json, inspect, traceback
fn = globals()[{entry_point!r}]
case_input = {repr(case_input)}

try:
    if isinstance(case_input, tuple):
        args = case_input
    elif isinstance(case_input, list):
         args = tuple(case_input)
    else:
        args = (case_input,)

    result = fn(*args)
    print(json.dumps({{"ok": True, "value": repr(result), "error": None}}))
except Exception:
    print(json.dumps({{"ok": False, "value": None, "error": traceback.format_exc()}}))
""".strip()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "case.py"
        path.write_text(program, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {timeout}s"

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None, completed.stderr.strip() or "no output"

    try:
        parsed = json.loads(lines[-1])
    except Exception:
        return None, short(completed.stdout + completed.stderr, 800)

    return parsed.get("value"), parsed.get("error")


def classify(original_status: Any, plus_status: Any, base_only: bool):
    original_ok = passed_status(original_status)
    plus_ok = True if base_only else passed_status(plus_status)

    if original_ok and plus_ok:
        return True, None, None

    if not original_ok:
        return False, "original_test_failure", "original"

    return False, "plus_test_failure", "plus"


def build_evaluation(problem, expected, solution, stage, failure_type, raw_part):
    if not stage or stage == "generation":
        return {
            "status": "failed",
            "failed_case_index": None,
            "input": None,
            "expected": None,
            "actual": None,
            "actual_error": None,
        }

    index = first_failed(raw_part)
    case_input = item_at(get_inputs(problem, stage), index)
    expected_output = item_at(get_expected(expected, stage), index)

    actual_output, actual_error = None, "failed input unavailable"
    if case_input is not None:
        actual_output, actual_error = run_actual(
            solution,
            problem.get("entry_point"),
            case_input,
        )

    return {
        "status": "failed",
        "failed_case_index": index,
        "input": repr(case_input) if case_input is not None else None,
        "expected": repr(expected_output) if expected_output is not None else None,
        "actual": actual_output,
        "actual_error": actual_error,
        "failure_type": failure_type,
    }


def explanation(passed: bool, generation: dict[str, Any], evaluation: dict[str, Any]):
    if passed:
        return "Passed all EvalPlus tests."

    if generation["status"] == "generation_failed":
        return f"Generation failed before evaluation: {generation['error']}."

    stage = "plus" if evaluation.get("failure_type") == "plus_test_failure" else "original"
    index = evaluation.get("failed_case_index")
    case_input = evaluation.get("input")
    expected = evaluation.get("expected")
    actual = evaluation.get("actual")
    actual_error = evaluation.get("actual_error")

    if expected is not None and actual is not None:
        return (
            f"Failed {stage} test case #{index}. "
            f"Input: {case_input}. Expected: {expected}. Actual: {actual}."
        )

    if actual_error:
        return (
            f"Failed {stage} test case #{index}. "
            f"Input: {case_input}. Actual output could not be computed because: "
            f"{short(actual_error, 500)}"
        )

    return (
        f"Failed {stage} test case #{index}. "
        "Expected and actual outputs were not available."
    )


def make_result(raw, base_only, problem, expected, solution, export_summary, skipped):
    task_id = str(raw["task_id"])
    method = str(export_summary.get("method", ""))

    original_raw = raw.get("base")
    plus_raw = None if base_only else raw.get("plus")

    original_status = part_status(original_raw)
    plus_status = None if base_only else part_status(plus_raw)

    passed, failure_type, stage = classify(original_status, plus_status, base_only)
    generation = generation_info(task_id, solution, export_summary, skipped)

    if skipped:
        passed = False
        failure_type = "generation_failed"
        stage = "generation"

    raw_part = original_raw if stage == "original" else plus_raw
    evaluation = build_evaluation(problem, expected, generation["code"], stage, failure_type, raw_part)
    evaluation["status"] = "passed" if passed else "failed"

    return {
        "task_id": task_id,
        "index": None,
        "entry_point": problem.get("entry_point"),
        "passed": passed,
        "failure_type": failure_type,
        "failure_explanation": explanation(passed, generation, evaluation),
        "task": task_info(task_id, problem, method),
        "generation": generation,
        "evaluation": evaluation,
    }


def eval_one(
    dataset,
    task_id,
    problem,
    solution,
    expected,
    base_only,
    test_details,
    min_time_limit,
    gt_time_limit_factor,
    export_summary,
    skipped,
):
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

        return make_result(raw, base_only, problem, expected, solution, export_summary, skipped)

    except Exception as exc:
        method = str(export_summary.get("method", ""))
        generation = generation_info(task_id, solution, export_summary, skipped)

        evaluation = {
            "status": "failed",
            "failed_case_index": None,
            "input": None,
            "expected": None,
            "actual": None,
            "actual_error": str(exc),
            "failure_type": "evaluation_error",
        }

        return {
            "task_id": task_id,
            "index": None,
            "entry_point": problem.get("entry_point"),
            "passed": False,
            "failure_type": "evaluation_error",
            "failure_explanation": f"EvalPlus evaluator crashed: {exc}",
            "task": task_info(task_id, problem, method),
            "generation": generation,
            "evaluation": evaluation,
        }


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
    problems, expected = load_problems(evalplus_dataset, mini, noextreme, version)
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
    results = []

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
                export_summary,
                skipped.get(task_id),
            ): task_id
            for task_id in task_ids
        }

        for future in as_completed(futures):
            row = future.result()
            row["index"] = index_of[row["task_id"]]
            results.append(row)

            status = "PASS" if row["passed"] else f"FAIL/{row['failure_type']}"
            print(f"[{row['index']}] {status} {row['task_id']}")

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
    passed_count = sum(1 for row in details if row.get("passed") is True)

    failures = Counter(
        row.get("failure_type") or "unknown_failure"
        for row in details
        if not row.get("passed")
    )

    score = passed_count / total if total else 0.0

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
        "passed": passed_count,
        "failed": total - passed_count,
        "pass@1": score,
        "pass@1_percent": round(score * 100, 2),
        "failure_counts": dict(failures),
    }