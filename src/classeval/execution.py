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
from src.classeval.core import JsonDict, class_name, task_id, text

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
    content = text(value)
    return content if len(content) <= limit else content[:limit] + "\n...[truncated]"

def imports_for(task: JsonDict) -> str:
    imports = task.get("import_statement") or []
    if isinstance(imports, str):
        return imports.strip()
    if isinstance(imports, list):
        return "\n".join(text(item) for item in imports if text(item))
    return ""

def test_classes(task: JsonDict) -> list[str]:
    values = task.get("test_classes") or []
    return [text(item) for item in values if text(item)] if isinstance(values, list) else []

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

def classify(stderr: str, returncode: int | None, *, timed_out: bool = False) -> str:
    if timed_out:
        return "timeout"
    checks = (
        ("SyntaxError", "syntax_error"),
        ("ModuleNotFoundError", "import_error"),
        ("ImportError", "import_error"),
        ("NameError", "name_error"),
        ("AttributeError", "attribute_error"),
        ("TypeError", "type_error"),
        ("KeyError", "key_error"),
        ("ValueError", "value_error"),
        ("AssertionError", "wrong_answer"),
        ("FAILED", "wrong_answer"),
        ("FAIL:", "wrong_answer"),
    )

    for marker, kind in checks:
        if marker in stderr:
            return kind
    return "runtime_error" if returncode not in (0, None) else "wrong_answer"

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
                "failure_type": "timeout",
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
    failures = errors = skipped = 0
    failed = re.search(r"FAILED\s+\(([^)]*)\)", output)

    if failed:
        for item in failed.group(1).split(","):
            key, _, value = item.strip().partition("=")
            if value.isdigit():
                if key == "failures":
                    failures = int(value)
                elif key == "errors":
                    errors = int(value)
                elif key == "skipped":
                    skipped = int(value)

    passed = None
    if total is not None:
        passed = total - skipped if result.get("passed") else total - failures - errors - skipped

    return {
        "total": total,
        "passed": passed,
        "failed": None if total is None or passed is None else total - passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
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

def failure_record(
    task: JsonDict,
    *,
    provider: str,
    model: str,
    stage: str,
    failure_type: str,
    error: str,
    generation_path: Path | None = None,
) -> JsonDict:
    return {
        **base_record(task, provider=provider, model=model, stage=stage),
        "passed": False,
        "failure_type": failure_type,
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
        return failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type="generation_failed",
            error=text(generation.get("error")) or "Generation failed.",
            generation_path=generation_path,
        )

    if not code:
        return failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type="empty_code",
            error="Generated code is empty.",
            generation_path=generation_path,
        )

    try:
        result = run(build_program(task, code), timeout)
    except Exception as exc:
        return failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type="evaluation_error",
            error=str(exc),
            generation_path=generation_path,
        )
    passed = result.get("passed") is True

    record = {
        **base_record(task, provider=provider, model=model, stage=stage),
        "passed": passed,
        "failure_type": None if passed else text(result.get("failure_type")) or "runtime_error",
        "error": None if passed else short(result.get("error"), 6000),
        "generation": {
            "path": str(generation_path),
            "stage": generation.get("stage"),
            "provider": generation.get("provider"),
            "model_name": generation.get("model_name"),
            "error": generation.get("error"),
        },
        "evaluation": {
            "returncode": result.get("returncode"),
            "stdout": short(result.get("stdout"), 3000),
            "stderr": short(result.get("stderr"), 6000),
            "error": short(result.get("error"), 6000),
            "failure_type": result.get("failure_type"),
            "test_classes": test_classes(task),
            "metrics": test_metrics(result),
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
        text(item.get("failure_type")) or "unknown"
        for item in results
        if item.get("passed") is not True
    )

    total_tests = passed_tests = 0
    for item in results:
        metrics = ((item.get("evaluation") or {}).get("metrics") or {})
        total_tests += metrics.get("total") or 0
        passed_tests += metrics.get("passed") or 0

    return {
        "benchmark": "ClassEval",
        "dataset": "classeval",
        "execution_model": "class_level",
        "stage": stage,
        "provider": provider,
        "model_name": model,
        "total_tasks": total,
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
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

__all__ = [
    "PRELUDE",
    "base_record",
    "build_program",
    "classify",
    "evaluate_generation",
    "failure_record",
    "imports_for",
    "run",
    "runner",
    "short",
    "summarize_results",
    "task_id",
    "test_classes",
    "test_metrics",
]