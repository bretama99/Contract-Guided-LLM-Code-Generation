import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


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
import ast

def extract_prompt_imports(prompt: str) -> str:
    imports = []
    seen = set()

    for line in prompt.splitlines():
        stripped = line.strip()

        if not (
            stripped.startswith("import ")
            or stripped.startswith("from ")
        ):
            continue

        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue

        if not all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
            continue

        if stripped not in seen:
            seen.add(stripped)
            imports.append(stripped)

    return "\n".join(imports)

def task_prompt(task: dict[str, Any]) -> str:
    for key in (
        "prompt",
        "complete_prompt",
        "code_prompt",
        "instruction",
        "description",
        "task_description",
    ):
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def task_entry_point(task: dict[str, Any]) -> str:
    for key in ("entry_point", "entrypoint", "function_name", "name"):
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def task_test_code(task: dict[str, Any]) -> str:
    for key in ("test", "test_code", "tests", "unit_tests"):
        value = task.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, list):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return "\n\n".join(parts)

    return ""


def build_humaneval_program(task: dict[str, Any], generated_code: str) -> str:
    prompt_imports = extract_prompt_imports(task.get("prompt", ""))
    test_code = task.get("test", "")
    entry_point = task.get("entry_point", "")

    return f"""
{COMMON_EVAL_IMPORTS}

{prompt_imports}

{generated_code}

{test_code}

check({entry_point})
""".strip()


def has_check_function(test_code: str) -> bool:
    return re.search(r"(?m)^\s*def\s+check\s*\(", test_code) is not None


def has_unittest_tests(test_code: str) -> bool:
    return "unittest.TestCase" in test_code


def has_pytest_style_tests(test_code: str) -> bool:
    return re.search(r"(?m)^\s*def\s+test_[A-Za-z_]\w*\s*\(", test_code) is not None


def bigcodebench_test_prefix(test_code: str, entry_point: str) -> str:
    if not entry_point:
        return ""

    if "candidate" in test_code and not re.search(r"(?m)^\s*candidate\s*=", test_code):
        return f"candidate = {entry_point}"

    return ""


def bigcodebench_test_suffix(test_code: str, entry_point: str) -> str:
    if not test_code.strip() or not entry_point:
        return ""

    if has_check_function(test_code):
        return f"check({entry_point})"

    if has_unittest_tests(test_code):
        return "unittest.main()"

    if has_pytest_style_tests(test_code):
        return """
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj):
        _obj()
""".strip()

    return ""


def build_bigcodebench_program(task: dict[str, Any], generated_code: str) -> str:
    prompt = task_prompt(task)
    prompt_imports = extract_prompt_imports(prompt)
    test_code = task_test_code(task)
    entry_point = task_entry_point(task)

    prefix = bigcodebench_test_prefix(test_code, entry_point)
    suffix = bigcodebench_test_suffix(test_code, entry_point)

    return f"""
{COMMON_EVAL_IMPORTS}

{prompt_imports}

{generated_code}

{prefix}

{test_code}

{suffix}
""".strip()


def classify_failure(stderr: str, error: str | None, timed_out: bool = False) -> str:
    combined = f"{stderr or ''}\n{error or ''}"

    if timed_out:
        return "timeout"
    if "SyntaxError" in combined:
        return "syntax_error"
    if "NameError" in combined:
        return "name_error"
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        return "import_error"
    if "AssertionError" in combined:
        return "wrong_answer"
    if error:
        return "runtime_error"

    return "runtime_error"

def run_python_program(program: str, timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_file = temp_path / "candidate_test.py"
        temp_file.write_text(program, encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("PYTHONNOUSERSITE", "1")

        try:
            completed = subprocess.run(
                [sys.executable, str(temp_file)],
                cwd=str(temp_path),   # CRITICAL: isolate candidate execution
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            passed = completed.returncode == 0
            stderr = completed.stderr or ""

            return {
                "passed": passed,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": stderr,
                "error": None if passed else stderr,
                "failure_type": None if passed else classify_failure(stderr, stderr),
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": f"Timeout after {timeout} seconds",
                "failure_type": "timeout",
            }

        except Exception as exc:
            error = str(exc)
            return {
                "passed": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error": error,
                "failure_type": classify_failure("", error),
            }
            
def evaluate_humaneval_candidate(
    task: dict[str, Any],
    generated_code: str,
    timeout: float,
) -> dict[str, Any]:
    program = build_humaneval_program(task, generated_code)
    return run_python_program(program, timeout)


def evaluate_bigcodebench_candidate(
    task: dict[str, Any],
    generated_code: str,
    timeout: float,
) -> dict[str, Any]:
    test_code = task_test_code(task)

    if not test_code.strip():
        return {
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "BigCodeBench task has no test, test_code, tests, or unit_tests field.",
            "failure_type": "missing_test_code",
        }

    program = build_bigcodebench_program(task, generated_code)
    return run_python_program(program, timeout)
