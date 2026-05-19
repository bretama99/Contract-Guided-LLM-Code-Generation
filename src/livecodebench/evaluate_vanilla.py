from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.config import OUTPUT_ROOT, RESULTS_ROOT, ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json
from src.common.task_utils import select_tasks


TASK_FILE = ROOT / "data" / "processed" / "livecodebench" / "livecodebench_tasks.json"

GEN_DIR = OUTPUT_ROOT / "vanilla"
EVAL_DIR = OUTPUT_ROOT / "livecodebench" / "evaluation" / "vanilla"
RESULT_DIR = RESULTS_ROOT / "livecodebench" / "vanilla"


def gen_path(task_id: str, provider: str, model: str) -> Path:
    return (
        GEN_DIR
        / safe_name(provider)
        / safe_name(model)
        / "livecodebench"
        / f"{safe_name(task_id)}_vanilla.json"
    )


def eval_path(task_id: str, provider: str, model: str) -> Path:
    return (
        EVAL_DIR
        / safe_name(provider)
        / safe_name(model)
        / f"{safe_name(task_id)}_eval.json"
    )


def norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).strip().splitlines())


def cases(task: dict[str, Any]) -> list[tuple[str, str]]:
    io = task.get("input_output") or {}
    return list(zip(map(str, io.get("inputs") or []), map(str, io.get("outputs") or [])))


def classify(stderr: str, returncode: int | None, timeout: bool = False) -> str:
    if timeout:
        return "timeout"
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "NameError" in stderr:
        return "name_error"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import_error"
    if returncode not in (0, None):
        return "runtime_error"
    return "wrong_answer"


def run_code(code: str, stdin: str, timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "main.py"
        path.write_text(code, encoding="utf-8")

        try:
            p = subprocess.run(
                [sys.executable, str(path)],
                input=stdin,
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": f"Timeout after {timeout}s",
                "failure_type": "timeout",
            }

    stderr = p.stderr or ""
    ok = p.returncode == 0

    return {
        "ok": ok,
        "returncode": p.returncode,
        "stdout": p.stdout or "",
        "stderr": stderr,
        "error": None if ok else stderr,
        "failure_type": None if ok else classify(stderr, p.returncode),
    }


def fail(task: dict[str, Any], generation: dict[str, Any] | None, kind: str, error: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "passed": False,
        "failure_type": kind,
        "failure_explanation": error,
        "generation": generation,
        "evaluation": {
            "status": "failed",
            "error": error,
        },
    }


def evaluate_one(task: dict[str, Any], provider: str, model: str, timeout: float) -> dict[str, Any]:
    path = gen_path(task["task_id"], provider, model)

    if not path.exists():
        return fail(task, None, "missing_generation", f"Missing generation file: {path}")

    generation = load_json(path)

    if generation.get("status") != "success":
        return fail(
            task,
            generation,
            "generation_error",
            str(generation.get("error") or "Generation failed."),
        )

    code = str(generation.get("generated_code") or "")

    if not code.strip():
        return fail(task, generation, "empty_code", "generated_code is empty.")

    test_cases = cases(task)

    if not test_cases:
        return fail(task, generation, "missing_test_cases", "No public test cases found.")

    for i, (stdin, expected) in enumerate(test_cases):
        result = run_code(code, stdin, timeout)

        if result["ok"] is not True:
            return {
                "task_id": task["task_id"],
                "passed": False,
                "failure_type": result["failure_type"],
                "failure_explanation": result["error"],
                "generation": generation,
                "evaluation": {
                    **result,
                    "case_index": i,
                    "input": stdin,
                    "expected": norm(expected),
                    "actual": norm(result["stdout"]),
                },
            }

        actual = norm(result["stdout"])
        expected_norm = norm(expected)

        if actual != expected_norm:
            return {
                "task_id": task["task_id"],
                "passed": False,
                "failure_type": "wrong_answer",
                "failure_explanation": f"Expected {expected_norm!r}, got {actual!r}",
                "generation": generation,
                "evaluation": {
                    "status": "failed",
                    "case_index": i,
                    "input": stdin,
                    "expected": expected_norm,
                    "actual": actual,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                },
            }

    return {
        "task_id": task["task_id"],
        "passed": True,
        "failure_type": None,
        "failure_explanation": "Passed all public stdin/stdout tests.",
        "generation": generation,
        "evaluation": {
            "status": "passed",
            "num_cases": len(test_cases),
        },
    }


def summarize(provider: str, model: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(r.get("passed") is True for r in results)
    failures = Counter(
        str(r.get("failure_type") or "unknown")
        for r in results
        if r.get("passed") is not True
    )

    missing_tests = failures.get("missing_test_cases", 0)
    testable = total - missing_tests

    return {
        "dataset": "livecodebench",
        "provider": provider,
        "model": model,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass@1_all": passed / total if total else 0,
        "testable_total": testable,
        "pass@1_testable": passed / testable if testable else 0,
        "failure_counts": dict(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LiveCodeBench vanilla generations")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)
    results = []

    for offset, task in enumerate(tasks):
        index = args.start + offset
        task_id = str(task["task_id"])
        out = eval_path(task_id, args.provider, args.model)

        if out.exists() and not args.overwrite:
            result = load_json(out)
        else:
            result = evaluate_one(task, args.provider, args.model, args.timeout)
            result["index"] = index
            save_json(out, result)

        result["index"] = index
        results.append(result)

        status = "PASS" if result.get("passed") else f"FAIL/{result.get('failure_type')}"
        print(f"[{index}] {status}: {task_id}")

    summary = summarize(args.provider, args.model, results)
    name = f"stage1_livecodebench_{safe_name(args.provider)}_{safe_name(args.model)}"

    save_json(RESULT_DIR / f"{name}_summary.json", summary)
    save_json(RESULT_DIR / f"{name}_details.json", results)

    print("\nLiveCodeBench evaluation complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()