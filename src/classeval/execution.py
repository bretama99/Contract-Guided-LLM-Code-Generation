from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.core import JsonDict, class_name, reference_leakage_report, task_id, text

FAILURE_SYNTAX = "syntax error"
FAILURE_IMPORT = "import error"
FAILURE_RUNTIME = "runtime error"
FAILURE_TIMEOUT = "timeout"
FAILURE_LOGICAL = "logical error"
FAILURE_MISSING_GENERATION = "missing generation"
FAILURE_GENERATION_FAILED = "generation failed"
FAILURE_EMPTY_GENERATION = "empty generation"
FAILURE_MISSING_METHOD = "missing method"
FAILURE_INCOMPLETE_METHOD = "incomplete method"
FAILURE_SIGNATURE = "signature error"
FAILURE_EVALUATION = "evaluation error"
FAILURE_INVALID_TEST_RESULT = "invalid test result"

ALLOWED_FAILURE_TYPES = {
    FAILURE_SYNTAX,
    FAILURE_IMPORT,
    FAILURE_RUNTIME,
    FAILURE_TIMEOUT,
    FAILURE_LOGICAL,
    FAILURE_MISSING_GENERATION,
    FAILURE_GENERATION_FAILED,
    FAILURE_EMPTY_GENERATION,
    FAILURE_MISSING_METHOD,
    FAILURE_INCOMPLETE_METHOD,
    FAILURE_SIGNATURE,
    FAILURE_EVALUATION,
    FAILURE_INVALID_TEST_RESULT,
}


def normalize_failure_type(value: Any) -> str:
    raw = text(value).lower().replace("_", " ").replace("-", " ")

    if raw in ALLOWED_FAILURE_TYPES:
        return raw

    if "missing generation" in raw or "missing generation file" in raw:
        return FAILURE_MISSING_GENERATION
    if "empty generation" in raw or "generated code is empty" in raw:
        return FAILURE_EMPTY_GENERATION
    if "incomplete method" in raw or "incomplete method bodies" in raw:
        return FAILURE_INCOMPLETE_METHOD
    if "missing method" in raw or "missing methods" in raw:
        return FAILURE_MISSING_METHOD
    if "signature" in raw or "staticmethod" in raw or "decorator" in raw:
        return FAILURE_SIGNATURE
    if "syntax" in raw or "indentation" in raw or "taberror" in raw or "invalid python" in raw:
        return FAILURE_SYNTAX
    if "import" in raw or "modulenotfounderror" in raw or "no module named" in raw:
        return FAILURE_IMPORT
    if "timeout" in raw or "timed out" in raw:
        return FAILURE_TIMEOUT
    if "logical" in raw or "wrong answer" in raw or "assert" in raw or "fail" in raw:
        return FAILURE_LOGICAL
    if "runtime" in raw or "exception" in raw or "typeerror" in raw or "valueerror" in raw or "attributeerror" in raw:
        return FAILURE_RUNTIME
    if "no tests were executed" in raw or "invalid test" in raw:
        return FAILURE_INVALID_TEST_RESULT
    if "generation" in raw:
        return FAILURE_GENERATION_FAILED

    return FAILURE_EVALUATION

PRELUDE = """
import unittest
import os
import sys
from typing import *

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
""".strip()


def short(value: Any, limit: int = 3000) -> str:
    value = text(value)
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"

def classify(stderr: str, returncode: int | None, *, timed_out: bool = False) -> str:
    if timed_out:
        return FAILURE_TIMEOUT

    stream = stderr or ""

    if "SyntaxError" in stream or "IndentationError" in stream or "TabError" in stream:
        return FAILURE_SYNTAX
    if "FAIL:" in stream or "AssertionError" in stream or re.search(r"FAILED \([^)]*failures=\d+", stream):
        return FAILURE_LOGICAL
    if returncode not in (0, None):
        return FAILURE_RUNTIME

    return FAILURE_INVALID_TEST_RESULT


def imports_for(task: JsonDict) -> str:
    value = task.get("import_statement") or []

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text(item) for item in value if text(item))

    return ""


def test_classes(task: JsonDict) -> list[str]:
    value = task.get("test_classes") or []
    return [text(item) for item in value if text(item)] if isinstance(value, list) else []


def runner(task: JsonDict) -> str:
    names = test_classes(task)

    if not names:
        return """
if __name__ == "__main__":
    unittest.main(argv=["ignored"], exit=True, verbosity=2)
""".strip()

    return f"""
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in {names!r}:
        cls = globals().get(name)
        if cls is None:
            raise NameError(f"Test class not found: {{name}}")
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
""".strip()


def build_program(task: JsonDict, code: str) -> str:
    test_code = text(task.get("test"))
    if not test_code:
        raise ValueError(f"{task_id(task)} has no test code.")

    return "\n\n".join(
        part for part in (PRELUDE, imports_for(task), code, test_code, runner(task)) if text(part)
    )


def run(program: str, timeout: float) -> JsonDict:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "candidate.py"
        path.write_text(program, encoding="utf-8")

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "failure_type": FAILURE_TIMEOUT,
                "error": f"Timeout after {timeout} seconds",
            }

    passed = completed.returncode == 0

    return {
        "passed": passed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "failure_type": None if passed else classify(completed.stderr, completed.returncode),
        "error": None if passed else completed.stderr,
    }


def test_metrics(result: JsonDict) -> JsonDict:
    output = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    total = int(match.group(1)) if match else None

    failures = 0
    errors = 0
    skipped = 0

    failed = re.search(r"FAILED\s+\(([^)]*)\)", output)
    if failed:
        for item in failed.group(1).split(","):
            key, _, value = item.strip().partition("=")
            if not value.isdigit():
                continue
            if key == "failures":
                failures = int(value)
            elif key == "errors":
                errors = int(value)
            elif key == "skipped":
                skipped = int(value)

    passed = None
    if isinstance(total, int):
        passed = total - skipped if result.get("passed") else max(0, total - failures - errors - skipped)

    return {
        "total": total,
        "passed": passed,
        "failed": None if total is None or passed is None else total - passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "executed": isinstance(total, int) and total > 0,
    }


def base_record(task: JsonDict, *, provider: str, model: str, stage: str) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": "ClassEval",
        "dataset": "classeval",
        "execution_model": "class_level",
        "class_name": class_name(task),
        "entry_point": class_name(task),
        "stage": stage,
        "provider": provider,
        "model_name": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def generation_metadata(generation: JsonDict, generation_path: Path) -> JsonDict:
    return {
        "path": str(generation_path),
        "stage": generation.get("stage"),
        "status": generation.get("status"),
        "provider": generation.get("provider"),
        "model_name": generation.get("model_name"),
        "contract_path": generation.get("contract_path"),
        "contract_type": generation.get("contract_type"),
        "contract_version": generation.get("contract_version"),
        "contract_status": generation.get("contract_status"),
        "error": generation.get("error"),
    }


def failure_record(
    task: JsonDict,
    *,
    provider: str,
    model: str,
    stage: str,
    failure_type: str,
    error: str,
    generation_path: Path | None = None,
    failure_stage: str = "evaluation",
) -> JsonDict:
    normalized = normalize_failure_type(failure_type)

    return {
        **base_record(task, provider=provider, model=model, stage=stage),
        "status": "failed",
        "passed": False,
        "failure_type": normalized,
        "failure_detail": text(failure_type) if text(failure_type) != normalized else None,
        "failure_stage": failure_stage,
        "error": error,
        "generation": {"path": str(generation_path)} if generation_path else None,
        "evaluation": None,
    }


def evaluate_generation(
    task: JsonDict,
    generation: JsonDict,
    generation_path: Path,
    *,
    provider: str,
    model: str,
    stage: str,
    timeout: float,
    include_code: bool = False,
) -> JsonDict:
    code = text(generation.get("generated_code") or generation.get("code"))

    if generation.get("status") != "success":
        record = failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type=normalize_failure_type(generation.get("error") or generation.get("failure_type") or FAILURE_GENERATION_FAILED),
            error=text(generation.get("error")) or "Generation failed.",
            generation_path=generation_path,
            failure_stage="generation",
        )
        record["generation"] = generation_metadata(generation, generation_path)
        return record

    if not code:
        record = failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type=normalize_failure_type(generation.get("error") or generation.get("failure_type") or FAILURE_GENERATION_FAILED),
            error="Generated code is empty.",
            generation_path=generation_path,
            failure_stage="generation",
        )
        record["generation"] = generation_metadata(generation, generation_path)
        return record

    leakage = reference_leakage_report(task, code, include_tests=False)

    try:
        result = run(build_program(task, code), timeout)
    except Exception as exc:
        record = failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type=normalize_failure_type(generation.get("error") or generation.get("failure_type") or FAILURE_GENERATION_FAILED),
            error=str(exc),
            generation_path=generation_path,
            failure_stage="evaluation",
        )
        record["generation"] = generation_metadata(generation, generation_path)
        record["reference_leakage"] = leakage
        return record

    metrics = test_metrics(result)
    tests_executed = metrics.get("executed") is True
    passed = result.get("passed") is True and tests_executed

    failure_type = None
    error = None

    if not passed:
        failure_type = normalize_failure_type(result.get("failure_type"))
        error = short(result.get("error"), 6000)

        if not tests_executed:
            failure_type = FAILURE_INVALID_TEST_RESULT
            error = "No tests were executed or unittest output did not report a test count."

    record = {
        **base_record(task, provider=provider, model=model, stage=stage),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "failure_type": failure_type,
        "error": error,
        "generation": generation_metadata(generation, generation_path),
        "contract_path": generation.get("contract_path"),
        "contract_type": generation.get("contract_type"),
        "contract_version": generation.get("contract_version"),
        "contract_status": generation.get("contract_status"),
        "reference_leakage": leakage,
        "evaluation": {
            "returncode": result.get("returncode"),
            "stdout": short(result.get("stdout"), 3000),
            "stderr": short(result.get("stderr"), 6000),
            "error": short(result.get("error"), 6000),
            "failure_type": result.get("failure_type"),
            "test_classes": test_classes(task),
            "metrics": metrics,
        },
    }

    if include_code:
        record["generated_code"] = code

    return record


def summarize_results(
    results: list[JsonDict],
    *,
    provider: str,
    model: str,
    stage: str,
    timeout: float,
) -> JsonDict:
    total = len(results)
    passed = sum(item.get("passed") is True for item in results)

    failures = Counter(
        normalize_failure_type(item.get("failure_type"))
        for item in results
        if item.get("passed") is not True
    )

    total_tests = 0
    passed_tests = 0
    executed_tasks = 0
    invalid_test_result_count = 0

    for item in results:
        metrics = ((item.get("evaluation") or {}).get("metrics") or {})
        total_for_item = metrics.get("total")
        passed_for_item = metrics.get("passed")

        if isinstance(total_for_item, int) and total_for_item > 0:
            executed_tasks += 1
            total_tests += total_for_item
            passed_tests += passed_for_item if isinstance(passed_for_item, int) else 0
        else:
            invalid_test_result_count += 1

    return {
        "benchmark": "ClassEval",
        "dataset": "classeval",
        "execution_model": "class_level",
        "stage": stage,
        "provider": provider,
        "model_name": model,
        "total_tasks": total,
        "executed_tasks": executed_tasks,
        "invalid_test_result_count": invalid_test_result_count,
        "passed": passed,
        "failed": total - passed,
        "pass@1": passed / total if total else 0.0,
        "pass@1_percent": round((passed / total * 100) if total else 0.0, 2),
        "timeout_seconds": timeout,
        "failure_counts": dict(failures),
        "tests": {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests else None,
            "computed_only_from_executed_tests": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "ALLOWED_FAILURE_TYPES",
    "FAILURE_LOGICAL",
    "FAILURE_RUNTIME",
    "FAILURE_SYNTAX",
    "FAILURE_TIMEOUT",
    "PRELUDE",
    "base_record",
    "build_program",
    "classify",
    "evaluate_generation",
    "failure_record",
    "generation_metadata",
    "imports_for",
    "normalize_failure_type",
    "run",
    "runner",
    "short",
    "summarize_results",
    "test_classes",
    "test_metrics",
]