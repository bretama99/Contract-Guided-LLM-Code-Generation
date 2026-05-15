import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common.io_utils import save_json

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "humaneval": ROOT / "data" / "processed" / "humaneval" / "humaneval_tasks.json",
    "bigcodebench": ROOT / "data" / "processed" / "bigcodebench" / "bigcodebench_tasks.json",
    "livecodebench": ROOT / "data" / "processed" / "livecodebench" / "livecodebench_tasks.json",
}

GEN_DIR = ROOT / "outputs" / "vanilla"
EVAL_DIR = ROOT / "outputs" / "evaluation" / "vanilla"
RESULT_DIR = ROOT / "results" / "vanilla"

PRELUDE = """
from typing import *
import math, re, sys, json, itertools, functools, collections, heapq, bisect
from collections import *

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
except Exception:
    pass
""".strip()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def short(value, limit: int = 3000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def classify(stderr: str, returncode: int) -> str:
    stderr = stderr or ""

    if returncode == 0:
        return "passed"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import_error"
    if "AssertionError" in stderr or "FAILED" in stderr:
        return "wrong_answer"
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "NameError" in stderr:
        return "name_error"
    return "runtime_error"


def build_program(code: str, task: dict) -> str:
    test = task.get("test")
    entry = task.get("entry_point")

    if not test:
        raise ValueError("No test field found for this task.")

    if "def check(" in test:
        runner = f"check({entry})"
    elif "unittest" in test or "TestCase" in test:
        runner = "import unittest\nunittest.main(argv=['ignored'], exit=True)"
    else:
        raise ValueError("Unsupported test format.")

    return (
        f"{PRELUDE}\n\n"
        f"# generated solution\n{code}\n\n"
        f"# benchmark test\n{test}\n\n"
        f"# run benchmark\n{runner}\n"
    )


def run_program(program: str, timeout: float) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "run.py"
        path.write_text(program, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(path)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def task_info(task: dict) -> dict:
    prompt = task.get("prompt") or task.get("instruct_prompt") or task.get("complete_prompt")

    return {
        "id": task.get("task_id"),
        "entry_point": task.get("entry_point"),
        "prompt": prompt,
        "test": task.get("test"),
    }


def generation_info(generation: dict | None) -> dict:
    generation = generation or {}

    return {
        "status": generation.get("status"),
        "error": generation.get("error"),
        "code": generation.get("generated_code"),
    }


def failed_assertion(stderr: str) -> str | None:
    for line in reversed((stderr or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("assert "):
            return stripped
    return None


def exception_signal(stderr: str = "", error: str | None = None) -> str | None:
    text = stderr or error or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    signals = []

    for line in lines:
        if (
            "AssertionError" in line
            or "SyntaxError" in line
            or "NameError" in line
            or "TypeError" in line
            or "ValueError" in line
            or "ModuleNotFoundError" in line
            or "ImportError" in line
            or "TimeoutExpired" in line
        ):
            signals.append(line)

    return "\n".join(signals[-3:]) if signals else None


def rewrite_candidate(expr: ast.AST, entry_point: str | None) -> str:
    if not entry_point:
        return ast.unparse(expr)

    class Rewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if node.id == "candidate":
                return ast.copy_location(ast.Name(id=entry_point, ctx=node.ctx), node)
            return node

    rewritten = Rewriter().visit(expr)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def parse_assertion(assertion: str, entry_point: str | None) -> tuple[str | None, str | None]:
    try:
        tree = ast.parse(assertion)
        node = tree.body[0]
    except Exception:
        return None, None

    if not isinstance(node, ast.Assert):
        return None, None

    test = node.test
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
    ):
        return (
            rewrite_candidate(test.left, entry_point),
            rewrite_candidate(test.comparators[0], entry_point),
        )

    return rewrite_candidate(test, entry_point), None


def compute_actual_expected(code: str, task: dict, assertion: str | None, timeout: float = 5.0) -> dict:
    if not assertion:
        return {
            "failed_assertion": None,
            "input": None,
            "expected": None,
            "actual": None,
            "actual_error": None,
        }

    lhs, rhs = parse_assertion(assertion, task.get("entry_point"))

    if not lhs or not rhs:
        return {
            "failed_assertion": assertion,
            "input": None,
            "expected": None,
            "actual": None,
            "actual_error": "Could not parse assertion into expected and actual expressions.",
        }

    program = f"""
{PRELUDE}

{code}

import json, traceback

try:
    actual = ({lhs})
    expected = ({rhs})
    print(json.dumps({{"ok": True, "actual": repr(actual), "expected": repr(expected), "error": None}}))
except Exception:
    print(json.dumps({{"ok": False, "actual": None, "expected": None, "error": traceback.format_exc()}}))
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
            return {
                "failed_assertion": assertion,
                "input": lhs,
                "expected": None,
                "actual": None,
                "actual_error": f"Timed out after {timeout}s while computing actual output.",
            }

    lines = completed.stdout.strip().splitlines()

    if not lines:
        return {
            "failed_assertion": assertion,
            "input": lhs,
            "expected": None,
            "actual": None,
            "actual_error": completed.stderr.strip() or "No output while computing actual output.",
        }

    try:
        parsed = json.loads(lines[-1])
    except Exception:
        return {
            "failed_assertion": assertion,
            "input": lhs,
            "expected": None,
            "actual": None,
            "actual_error": short(completed.stdout + completed.stderr, 800),
        }

    return {
        "failed_assertion": assertion,
        "input": lhs,
        "expected": parsed.get("expected"),
        "actual": parsed.get("actual"),
        "actual_error": parsed.get("error"),
    }


def evaluation_info(
    *,
    passed: bool,
    failure_type: str | None,
    code: str | None,
    task: dict,
    returncode=None,
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
    timed_out: bool = False,
) -> dict:
    case = compute_actual_expected(code or "", task, failed_assertion(stderr))

    return {
        "status": "passed" if passed else "failed",
        "returncode": returncode,
        "stdout": short(stdout, 3000),
        "stderr": short(stderr, 6000),
        "timed_out": timed_out,
        "failure_type": failure_type,
        "failed_assertion": case["failed_assertion"],
        "input": case["input"],
        "expected": case["expected"],
        "actual": case["actual"],
        "actual_error": case["actual_error"],
        "exception": exception_signal(stderr, error),
    }


def failure_explanation(passed: bool, failure_type: str | None, generation: dict, evaluation: dict) -> str:
    if passed:
        return "Passed all evaluated benchmark tests."

    if failure_type == "generation_error":
        return f"Generation failed before evaluation: {generation.get('error')}."

    if failure_type == "empty_code":
        return "Generation failed because generated_code is empty."

    if failure_type == "timeout":
        return "The generated code did not finish within the time limit."

    if evaluation.get("expected") is not None and evaluation.get("actual") is not None:
        return (
            f"Failed test case. Input: {evaluation.get('input')}. "
            f"Expected: {evaluation.get('expected')}. "
            f"Actual: {evaluation.get('actual')}."
        )

    if evaluation.get("actual_error"):
        return (
            f"Failed test case. Input: {evaluation.get('input')}. "
            f"Actual output could not be computed because: {short(evaluation.get('actual_error'), 500)}"
        )

    if evaluation.get("failed_assertion"):
        return f"Failed assertion: {evaluation.get('failed_assertion')}."

    if evaluation.get("exception"):
        return f"Failed with exception: {evaluation.get('exception')}."

    if failure_type:
        return f"Failed with failure type: {failure_type}."

    return "Failed evaluation."


def make_result(task: dict, generation: dict, passed: bool, failure_type: str | None, evaluation: dict) -> dict:
    gen = generation_info(generation)
    return {
        "task_id": str(task.get("task_id")),
        "index": None,
        "entry_point": task.get("entry_point"),
        "passed": passed,
        "failure_type": failure_type,
        "failure_explanation": failure_explanation(passed, failure_type, gen, evaluation),
        "task": task_info(task),
        "generation": gen,
        "evaluation": evaluation,
    }


def evaluate_file(gen_file: Path, task: dict, timeout: float, retry_timeout: float) -> dict:
    generation = load_json(gen_file)

    if generation.get("status") != "success":
        failure_type = "generation_error"
        evaluation = evaluation_info(
            passed=False,
            failure_type=failure_type,
            code=None,
            task=task,
            error=str(generation.get("error") or "Generation failed."),
        )
        return make_result(task, generation, False, failure_type, evaluation)

    code = generation.get("generated_code")

    if not code:
        failure_type = "empty_code"
        evaluation = evaluation_info(
            passed=False,
            failure_type=failure_type,
            code=None,
            task=task,
            error="generated_code is empty",
        )
        return make_result(task, generation, False, failure_type, evaluation)

    try:
        program = build_program(code, task)
    except Exception as error:
        failure_type = "evaluation_error"
        evaluation = evaluation_info(
            passed=False,
            failure_type=failure_type,
            code=code,
            task=task,
            error=str(error),
        )
        return make_result(task, generation, False, failure_type, evaluation)

    try:
        completed = run_program(program, timeout)
    except subprocess.TimeoutExpired:
        if retry_timeout and retry_timeout > timeout:
            try:
                completed = run_program(program, retry_timeout)
            except subprocess.TimeoutExpired:
                failure_type = "timeout"
                evaluation = evaluation_info(
                    passed=False,
                    failure_type=failure_type,
                    code=code,
                    task=task,
                    error=f"Timed out after retry timeout of {retry_timeout} seconds",
                    timed_out=True,
                )
                return make_result(task, generation, False, failure_type, evaluation)
        else:
            failure_type = "timeout"
            evaluation = evaluation_info(
                passed=False,
                failure_type=failure_type,
                code=code,
                task=task,
                error=f"Timed out after {timeout} seconds",
                timed_out=True,
            )
            return make_result(task, generation, False, failure_type, evaluation)
    except Exception as error:
        failure_type = "evaluation_error"
        evaluation = evaluation_info(
            passed=False,
            failure_type=failure_type,
            code=code,
            task=task,
            error=str(error),
        )
        return make_result(task, generation, False, failure_type, evaluation)

    failure_type = classify(completed.stderr, completed.returncode)
    passed = completed.returncode == 0

    evaluation = evaluation_info(
        passed=passed,
        failure_type=failure_type,
        code=code,
        task=task,
        returncode=completed.returncode,
        stdout=completed.stdout[-3000:],
        stderr=completed.stderr[-6000:],
    )

    return make_result(task, generation, passed, failure_type, evaluation)

def summarize(dataset: str, provider: str, model: str, results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item.get("passed") is True)
    failures = Counter(item.get("failure_type", "unknown") for item in results)

    import_errors = failures.get("import_error", 0)
    timeouts = failures.get("timeout", 0)

    runnable_total = total - import_errors
    executed_total = total - import_errors - timeouts

    return {
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass@1": passed / total if total else 0,
        "runnable_total": runnable_total,
        "runnable_pass@1": passed / runnable_total if runnable_total else 0,
        "executed_total": executed_total,
        "executed_pass@1": passed / executed_total if executed_total else 0,
        "failure_counts": dict(failures),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vanilla generated code")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--retry-timeout", type=float, default=45)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset == "livecodebench":
        print("LiveCodeBench is not supported by this simple evaluator yet.")
        print("Use this evaluator only for HumanEval and BigCodeBench.")
        return

    tasks = {str(task["task_id"]): task for task in load_json(DATASETS[args.dataset])}
    gen_dir = GEN_DIR / safe(args.provider) / safe(args.model) / args.dataset
    gen_files = sorted(gen_dir.glob("*_vanilla.json"))

    if args.count:
        gen_files = gen_files[: args.count]

    if not gen_files:
        print(f"No generated files found in: {gen_dir}")
        return

    def process_one(index: int, gen_file: Path):
        generation = load_json(gen_file)
        task_id = str(generation.get("task_id"))
        task = tasks.get(task_id)

        if not task:
            return index, None, f"[{index}] SKIP task not found: {task_id}"

        eval_file = (
            EVAL_DIR
            / safe(args.provider)
            / safe(args.model)
            / args.dataset
            / f"{safe(task_id)}_eval.json"
        )

        if eval_file.exists() and not args.overwrite:
            result = load_json(eval_file)
        else:
            result = evaluate_file(gen_file, task, args.timeout, args.retry_timeout)
            result["index"] = index
            save_json(eval_file, result)

        result["index"] = index
        status = "PASS" if result.get("passed") else f"FAIL/{result.get('failure_type')}"
        return index, result, f"[{index}] {status}: {task_id}"

    indexed_results = []

    if args.workers <= 1:
        for index, gen_file in enumerate(gen_files):
            item = process_one(index, gen_file)
            print(item[2])
            if item[1]:
                indexed_results.append(item)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(process_one, index, gen_file)
                for index, gen_file in enumerate(gen_files)
            ]

            for future in as_completed(futures):
                item = future.result()
                print(item[2])
                if item[1]:
                    indexed_results.append(item)

    indexed_results.sort(key=lambda item: item[0])
    results = [item[1] for item in indexed_results]

    summary = summarize(args.dataset, args.provider, args.model, results)
    name = f"stage1_{args.dataset}_{args.provider}_{safe(args.model)}"

    save_json(RESULT_DIR / f"{name}_summary.json", summary)
    save_json(RESULT_DIR / f"{name}_details.json", results)

    print("\nEvaluation complete")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()