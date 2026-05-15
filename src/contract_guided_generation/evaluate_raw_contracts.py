import argparse
import ast
import json
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any

from src.common.config import DATASETS
from src.common.execution import evaluate_humaneval_candidate, evaluate_bigcodebench_candidate
from src.common.io_utils import load_json, load_json_list, save_json
from src.common.llm_clients import default_model
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_results_folder
from src.common.task_utils import select_tasks, task_identifier, task_prompt

STAGE = "stage_2c_raw_contract_guided_evaluation"
METHOD = "raw_contract_guided_generation"

PRELUDE = """
from typing import *
import math, re, sys, json, itertools, functools, collections, heapq, bisect
from collections import *
""".strip()


def short(value: Any, limit: int = 3000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def signal(text: str = "") -> str | None:
    keys = (
        "AssertionError", "SyntaxError", "NameError", "TypeError", "ValueError",
        "ModuleNotFoundError", "ImportError", "TimeoutExpired",
    )
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    hits = [line for line in lines if line.startswith("assert ") or any(k in line for k in keys)]
    return "\n".join(hits[-3:]) if hits else None


def failed_assertion(stderr: str) -> str | None:
    for line in reversed(str(stderr or "").splitlines()):
        if line.strip().startswith("assert "):
            return line.strip()
    return None


def rewrite_candidate(expr: ast.AST, entry_point: str | None) -> str:
    if not entry_point:
        return ast.unparse(expr)

    class Rewriter(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            return ast.copy_location(ast.Name(entry_point, node.ctx), node) if node.id == "candidate" else node

    rewritten = Rewriter().visit(expr)
    ast.fix_missing_locations(rewritten)
    return ast.unparse(rewritten)


def parse_assertion(assertion: str, entry_point: str | None) -> tuple[str | None, str | None]:
    try:
        node = ast.parse(assertion).body[0]
        test = node.test if isinstance(node, ast.Assert) else None
    except Exception:
        return None, None

    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        return rewrite_candidate(test.left, entry_point), rewrite_candidate(test.comparators[0], entry_point)

    return (rewrite_candidate(test, entry_point), None) if test else (None, None)


def actual_expected(code: str, task: dict[str, Any], assertion: str | None) -> dict[str, Any]:
    if not assertion:
        return {"failed_assertion": None, "input": None, "expected": None, "actual": None, "actual_error": None}

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
    print(json.dumps({{"actual": repr(actual), "expected": repr(expected), "error": None}}))
except Exception:
    print(json.dumps({{"actual": None, "expected": None, "error": traceback.format_exc()}}))
""".strip()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = f"{temp_dir}/case.py"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)

        try:
            completed = subprocess.run(
                [sys.executable, path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return {
                "failed_assertion": assertion,
                "input": lhs,
                "expected": None,
                "actual": None,
                "actual_error": "Timed out while computing actual output.",
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
        parsed = {"actual": None, "expected": None, "error": short(completed.stdout + completed.stderr, 800)}

    return {
        "failed_assertion": assertion,
        "input": lhs,
        "expected": parsed.get("expected"),
        "actual": parsed.get("actual"),
        "actual_error": parsed.get("error"),
    }


def task_info(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(task.get("task_id")),
        "entry_point": task.get("entry_point"),
        "prompt": task_prompt(task),
        "test": task.get("test"),
    }


def load_contract(generation: dict[str, Any]) -> Any:
    path = generation.get("contract_path")
    if not path:
        return None
    try:
        record = load_json(path)
    except Exception:
        return None
    return record.get("contract", record) if isinstance(record, dict) else None


def generation_info(generation: dict[str, Any] | None) -> dict[str, Any]:
    generation = generation or {}
    return {
        "status": generation.get("status"),
        "error": generation.get("error"),
        "code": generation.get("generated_code"),
        "contract": load_contract(generation),
    }


def evaluation_info(
    *,
    passed: bool,
    failure_type: str | None,
    code: str | None,
    task: dict[str, Any],
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    stderr = result.get("stderr", "")
    case = actual_expected(code or "", task, failed_assertion(stderr))

    return {
        "status": "passed" if passed else "failed",
        "returncode": result.get("returncode"),
        "stdout": short(result.get("stdout", "")),
        "stderr": short(stderr, 6000),
        "failure_type": failure_type,
        "failed_assertion": case["failed_assertion"],
        "input": case["input"],
        "expected": case["expected"],
        "actual": case["actual"],
        "actual_error": case["actual_error"],
        "exception": signal(stderr or error or ""),
    }


def explain(passed: bool, failure_type: str | None, generation: dict[str, Any], evaluation: dict[str, Any]) -> str:
    if passed:
        return "Passed all evaluated benchmark tests."

    if failure_type in {"missing_generation", "invalid_generation_file", "generation_failed", "empty_generated_code"}:
        return f"Generation failed before evaluation: {generation.get('error')}."

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

    return f"Failed with failure type: {failure_type}."


def make_result(task: dict[str, Any], generation: dict[str, Any] | None, passed: bool, failure_type: str | None, evaluation: dict[str, Any]) -> dict[str, Any]:
    gen = generation_info(generation)
    return {
        "task_id": str(task.get("task_id")),
        "index": None,
        "entry_point": task.get("entry_point"),
        "passed": passed,
        "failure_type": failure_type,
        "failure_explanation": explain(passed, failure_type, gen, evaluation),
        "task": task_info(task),
        "generation": gen,
        "evaluation": evaluation,
    }


def fail(task: dict[str, Any], kind: str, error: str) -> dict[str, Any]:
    generation = {"status": kind, "error": error, "generated_code": None}
    evaluation = evaluation_info(passed=False, failure_type=kind, code=None, task=task, error=error)
    return make_result(task, generation, False, kind, evaluation)


def evaluate_one(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task_id = task_identifier(task)
    gen_file = raw_contract_code_path(args.dataset, task_id, args.provider, args.model)

    if not gen_file.exists():
        return fail(task, "missing_generation", f"Missing generated file: {gen_file}")

    generation = load_json(gen_file)
    if not isinstance(generation, dict):
        return fail(task, "invalid_generation_file", "Generation file is not a JSON object.")

    if generation.get("status") != "success":
        return fail(task, "generation_failed", str(generation.get("error")))

    code = generation.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        return fail(task, "empty_generated_code", "Generated code is empty or missing.")

    result = evaluate_candidate(args.dataset, task, code, args.timeout)
    passed = result.get("passed") is True
    failure_type = result.get("failure_type")

    evaluation = evaluation_info(
        passed=passed,
        failure_type=failure_type,
        code=code,
        task=task,
        result=result,
        error=result.get("error"),
    )

    return make_result(task, generation, passed, failure_type, evaluation)


def evaluate_candidate(dataset: str, task: dict[str, Any], code: str, timeout: float) -> dict[str, Any]:
    if "evalplus_dataset" in DATASETS[dataset]:
        raise SystemExit("EvalPlus datasets must be evaluated with: python scripts/evaluate_evalplus.py --method raw_contracts")

    if dataset == "humaneval":
        return evaluate_humaneval_candidate(task, code, timeout)

    if dataset == "bigcodebench":
        return evaluate_bigcodebench_candidate(task, code, timeout)

    raise NotImplementedError(f"No raw-contract evaluator implemented for dataset: {dataset}")


def summarize(benchmark: str, args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(row.get("passed") is True for row in results)
    failures = Counter(str(row.get("failure_type", "unknown")) for row in results if row.get("passed") is not True)
    pass_at_1 = passed / total if total else 0.0

    return {
        "benchmark": benchmark,
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model,
        "method": METHOD,
        "stage": STAGE,
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "missing": failures.get("missing_generation", 0),
        "pass@1": pass_at_1,
        "pass@1_percent": round(pass_at_1 * 100, 2),
        "timeout_seconds": args.timeout,
        "failure_counts": dict(failures),
    }


def evaluate(args: argparse.Namespace) -> None:
    if "evalplus_dataset" in DATASETS[args.dataset]:
        raise SystemExit("Use scripts/evaluate_evalplus.py --method raw_contracts for EvalPlus datasets.")

    args.model = args.model or default_model(args.provider)
    info = DATASETS[args.dataset]
    tasks = select_tasks(load_json_list(info["path"]), start=args.start, count=args.count)
    results = []

    for index, task in enumerate(tasks, start=args.start):
        result = evaluate_one(task, args)
        result["index"] = index
        results.append(result)

        status = "PASS" if result.get("passed") is True else f"FAIL/{result.get('failure_type', 'unknown')}"
        print(f"[{index}] {status} {result['task_id']}")

    summary = summarize(info["label"], args, results)
    out_dir = raw_contract_results_folder(args.dataset, args.provider, args.model)

    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "details.json", results)

    print("\nStage 2C raw contract-guided evaluation finished")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2C: evaluate raw contract-guided generations")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="humaneval")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())