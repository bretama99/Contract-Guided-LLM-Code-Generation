from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

FAILURE_SYNTAX = "syntax error"
FAILURE_RUNTIME = "runtime error"
FAILURE_TIMEOUT = "timeout"
FAILURE_LOGICAL = "logical error"
FAILURE_OTHER = "other"

COMMON_EVAL_IMPORTS = """
from typing import *
import math
import re
import itertools
import collections
import functools
import heapq
import bisect
import unittest
""".strip()


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_failure_type(value: Any) -> str:
    value = text(value).lower().replace("_", " ").replace("-", " ")

    if value in {FAILURE_SYNTAX, FAILURE_RUNTIME, FAILURE_TIMEOUT, FAILURE_LOGICAL, FAILURE_OTHER}:
        return value
    if "syntax" in value or "indentation" in value or "taberror" in value:
        return FAILURE_SYNTAX
    if "timeout" in value or "timed out" in value:
        return FAILURE_TIMEOUT
    if "assert" in value or "wrong answer" in value or "fail" in value or "logical" in value:
        return FAILURE_LOGICAL
    if "runtime" in value or "exception" in value or "error" in value:
        return FAILURE_RUNTIME

    return FAILURE_OTHER


def classify_failure(stderr: str = "", error: str | None = None, timed_out: bool = False) -> str:
    if timed_out:
        return FAILURE_TIMEOUT

    stream = f"{stderr or ''}\n{error or ''}"

    if "SyntaxError" in stream or "IndentationError" in stream or "TabError" in stream:
        return FAILURE_SYNTAX
    if "AssertionError" in stream or "FAIL:" in stream or re.search(r"FAILED\s+\(", stream):
        return FAILURE_LOGICAL
    if stream.strip():
        return FAILURE_RUNTIME

    return FAILURE_OTHER


def extract_prompt_imports(prompt: str) -> str:
    imports: list[str] = []
    seen: set[str] = set()

    for line in prompt.splitlines():
        line = line.strip()
        if not (line.startswith("import ") or line.startswith("from ")):
            continue

        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue

        if all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body) and line not in seen:
            seen.add(line)
            imports.append(line)

    return "\n".join(imports)


def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "complete_prompt", "code_prompt", "instruction", "description", "task_description"):
        value = text(task.get(key))
        if value:
            return value
    return ""


def task_entry_point(task: dict[str, Any]) -> str:
    for key in ("entry_point", "entrypoint", "function_name", "name"):
        value = text(task.get(key))
        if value:
            return value
    return ""


def task_test_code(task: dict[str, Any]) -> str:
    for key in ("test", "test_code", "tests", "unit_tests"):
        value = task.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, list):
            parts = [text(item) for item in value if text(item)]
            if parts:
                return "\n\n".join(parts)

    return ""


def build_humaneval_program(task: dict[str, Any], generated_code: str) -> str:
    prompt = task.get("prompt", "")
    test_code = text(task.get("test"))
    entry_point = task_entry_point(task)

    if not test_code:
        raise ValueError("HumanEval task has no test field.")
    if not entry_point:
        raise ValueError("HumanEval task has no entry point.")

    return "\n\n".join(
        part
        for part in (
            COMMON_EVAL_IMPORTS,
            extract_prompt_imports(prompt),
            generated_code,
            test_code,
            f"check({entry_point})",
        )
        if text(part)
    )


def has_check_function(test_code: str) -> bool:
    return re.search(r"(?m)^\s*def\s+check\s*\(", test_code) is not None


def has_unittest_tests(test_code: str) -> bool:
    return "unittest.TestCase" in test_code


def has_pytest_style_tests(test_code: str) -> bool:
    return re.search(r"(?m)^\s*def\s+test_[A-Za-z_]\w*\s*\(", test_code) is not None


def bigcodebench_test_prefix(test_code: str, entry_point: str) -> str:
    if entry_point and "candidate" in test_code and not re.search(r"(?m)^\s*candidate\s*=", test_code):
        return f"candidate = {entry_point}"
    return ""


def bigcodebench_test_suffix(test_code: str, entry_point: str) -> str:
    if not test_code or not entry_point:
        return ""
    if has_check_function(test_code):
        return f"check({entry_point})"
    if has_unittest_tests(test_code):
        return "unittest.main(argv=['ignored'], exit=True)"
    if has_pytest_style_tests(test_code):
        return """
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj):
        _obj()
""".strip()
    return ""


def build_bigcodebench_program(task: dict[str, Any], generated_code: str) -> str:
    prompt = task_prompt(task)
    test_code = task_test_code(task)
    entry_point = task_entry_point(task)

    if not test_code:
        raise ValueError("BigCodeBench task has no test/test_code/tests/unit_tests field.")

    return "\n\n".join(
        part
        for part in (
            COMMON_EVAL_IMPORTS,
            extract_prompt_imports(prompt),
            generated_code,
            bigcodebench_test_prefix(test_code, entry_point),
            test_code,
            bigcodebench_test_suffix(test_code, entry_point),
        )
        if text(part)
    )


def run_python_program(program: str, timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "candidate_test.py"
        path.write_text(program, encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

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
                "error": f"Timeout after {timeout} seconds",
                "failure_type": FAILURE_TIMEOUT,
            }
        except Exception as exc:
            error = str(exc)
            return {
                "passed": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": error,
                "failure_type": classify_failure(error=error),
            }

    passed = completed.returncode == 0
    stderr = completed.stderr or ""

    return {
        "passed": passed,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": stderr,
        "error": None if passed else stderr,
        "failure_type": None if passed else classify_failure(stderr=stderr, error=stderr),
    }


def evaluate_humaneval_candidate(task: dict[str, Any], generated_code: str, timeout: float) -> dict[str, Any]:
    return run_python_program(build_humaneval_program(task, generated_code), timeout)


def evaluate_bigcodebench_candidate(task: dict[str, Any], generated_code: str, timeout: float) -> dict[str, Any]:
    return run_python_program(build_bigcodebench_program(task, generated_code), timeout)