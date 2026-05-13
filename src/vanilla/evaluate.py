import argparse
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

def base_result(gen_file: Path, generation: dict) -> dict:
    return {
        "task_id": generation.get("task_id"),
        "benchmark": generation.get("benchmark"),
        "provider": generation.get("provider"),
        "model_name": generation.get("model_name"),
        "stage": "vanilla",
        "generation_file": str(gen_file),
    }

def evaluate_file(gen_file: Path, task: dict, timeout: float, retry_timeout: float) -> dict:
    generation = load_json(gen_file)
    result = base_result(gen_file, generation)
    if generation.get("status") != "success":
        result.update(
            passed=False,
            failure_type="generation_error",
            error=generation.get("error"),
        )
        return result

    code = generation.get("generated_code")
    if not code:
        result.update(
            passed=False,
            failure_type="empty_code",
            error="generated_code is empty",
        )
        return result

    try:
        program = build_program(code, task)
    except Exception as error:
        failure_type = "runtime_error"
        result.update(
            passed=False,
            failure_type=failure_type,
            error=str(error),
        )
        return result
    try:
        completed = run_program(program, timeout)
        
    except subprocess.TimeoutExpired:
        if retry_timeout and retry_timeout > timeout:
            try:
                completed = run_program(program, retry_timeout)
            except subprocess.TimeoutExpired:
                result.update(
                    passed=False,
                    failure_type="timeout",
                    timed_out=True,
                    error=f"Timed out after retry timeout of {retry_timeout} seconds",
                )
                return result
        else:
            result.update(
                passed=False,
                failure_type="timeout",
                timed_out=True,
                error=f"Timed out after {timeout} seconds",
            )
            return result
    except Exception as error:
        result.update(
            passed=False,
            failure_type="evaluation_error",
            error=str(error),
        )
        return result

    result.update(
        passed=completed.returncode == 0,
        failure_type=classify(completed.stderr, completed.returncode),
        returncode=completed.returncode,
        stdout=completed.stdout[-1000:],
        stderr=completed.stderr[-3000:],
    )
    return result

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

    tasks = {
        str(task["task_id"]): task
        for task in load_json(DATASETS[args.dataset])
    }

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
            save_json(eval_file, result)

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