from __future__ import annotations
import argparse
import inspect
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, namedtuple
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def _getargspec(fn: Any) -> Any:
        s = inspect.getfullargspec(fn)
        return _ArgSpec(s.args, s.varargs, s.varkw, s.defaults)
    inspect.getargspec = _getargspec
from src.generation.generation_core import (
    load_tasks,
    read_json,
    save_json,
    selected_tasks,
    taco_io,
    task_id,
)
from src.generation import testing_util as taco
RAW_TYPES = {
    1: "passed",
    0: "logical_error",
    -1: "timeout",
    -2: "compile_or_load_error",
    -3: "runtime_or_process_error",
    -4: "compile_error",
}
PRIORITY = (
    "syntax_error",
    "compile_error",
    "compile_or_load_error",
    "runtime_or_process_error",
    "timeout",
    "logical_error",
    "incomplete_results",
    "unknown_result",
)
def index_code(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Code directory not found: {root}")
    indexed: dict[str, Path] = {}
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("_"):
            continue
        ids = {path.stem}
        try:
            obj = read_json(path)
            stored = obj.get("task_id") if isinstance(obj, dict) else None
            if isinstance(stored, str) and stored.strip():
                ids.add(stored.strip())
        except Exception:
            pass
        path = path.resolve()
        for current_id in ids:
            old = indexed.get(current_id)
            if old is not None and old != path:
                raise ValueError(
                    f"Duplicate code for {current_id}:\n- {old}\n- {path}"
                )
            indexed[current_id] = path
    return indexed

def load_code(path: Path) -> str:
    obj = read_json(path)
    code = obj.get("code") if isinstance(obj, dict) else None
    if not isinstance(code, str) or not code.strip():
        raise ValueError("missing non-empty 'code' field")
    return code.strip()

def prepare_task(task: dict, limit: int | None) -> tuple[dict, int]:
    io = dict(taco_io(task))
    inputs, outputs = io.get("inputs"), io.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("input_output requires inputs and outputs lists")
    total = min(len(inputs), len(outputs))
    if limit is not None:
        total = min(total, limit)
        io["inputs"], io["outputs"] = inputs[:total], outputs[:total]
    sample = dict(task)
    sample["input_output"] = json.dumps(io, ensure_ascii=False)
    return sample, total

def normalize(value: Any) -> Any:
    try:
        return value.item() if hasattr(value, "item") else value
    except Exception:
        return value

def outcome(value: Any) -> str:
    value = normalize(value)
    if value is True:
        return "passed"
    if value is False:
        return "logical_error"
    return RAW_TYPES.get(value, "unknown_result") if type(value) is int else "unknown_result"

def classify(results: list[Any], total: int) -> tuple[str | None, list[str], dict[str, int]]:
    counts = Counter(outcome(v) for v in results)
    if total > 0 and len(results) == total and counts["passed"] == total:
        return None, [], dict(counts)
    reasons = [name for name in PRIORITY if counts.get(name)]
    if len(results) != total:
        reasons.append("incomplete_results")
    reasons = list(dict.fromkeys(reasons))
    primary = next((name for name in PRIORITY if name in reasons), "unknown_result")
    return primary, reasons, dict(counts)

def failed(task_id_value: str, total: int, kind: str, path: str | None, error: str) -> dict:
    return {
        "task_id": task_id_value,
        "passed": False,
        "tests_run": 0,
        "tests_total": total,
        "tests_passed": 0,
        "pass_percentage": 0.0,
        "failure_type": kind,
        "failure_reasons": [kind],
        "outcome_counts": {kind: 1},
        "raw_results": [],
        "error": error or None,
        "code_path": path,
    }

def evaluate(payload: tuple) -> tuple[int, dict]:
    position, current_id, sample, total, code_path, timeout, debug = payload
    workdir = tempfile.mkdtemp(prefix=f"taco_eval_{current_id}_")
    try:
        try:
            code = load_code(Path(code_path))
        except Exception as exc:
            record = failed(
                current_id, total, "invalid_code_file", code_path, repr(exc)
            )
        else:
            try:
                compile(code, "<generated_solution>", "exec")
            except SyntaxError as exc:
                record = failed(
                    current_id, total, "syntax_error", code_path,
                    f"{exc.msg}; line={exc.lineno}; column={exc.offset}",
                )
            else:
                os.chdir(workdir)
                taco.TIMEOUT = int(timeout)
                if not debug:
                    sink = open(os.devnull, "w")
                    os.dup2(sink.fileno(), 1)
                    os.dup2(sink.fileno(), 2)
                raw = taco.run_test(sample, test=code, debug=debug)
                if not isinstance(raw, list):
                    raise TypeError(f"run_test returned {type(raw).__name__}, not list")
                results = [normalize(v) for v in raw]
                primary, reasons, counts = classify(results, total)
                tests_passed = counts.get("passed", 0)
                record = {
                    "task_id": current_id,
                    "passed": primary is None,
                    "tests_run": len(results),
                    "tests_total": total,
                    "tests_passed": tests_passed,
                    "pass_percentage": round(100 * tests_passed / total, 4),
                    "failure_type": primary,
                    "failure_reasons": reasons,
                    "outcome_counts": counts,
                    "raw_results": results,
                    "error": None,
                    "code_path": code_path,
                }
    except BaseException as exc:
        record = failed(
            current_id, total, "evaluator_error", code_path,
            f"{type(exc).__name__}: {exc}",
        )
    record["_workdir"] = workdir
    return position, record

def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate local code with local TACO.")
    p.add_argument("--task-file", type=Path, required=True)
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--workflow", choices=("no_solution", "with_solution"), required=True)
    p.add_argument("--model-name", default="local-model")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int)
    p.add_argument("--timeout", type=int, default=4, help="Seconds per test.")
    p.add_argument("--limit-tests", type=int)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    if not args.task_file.is_file() or not args.code_dir.is_dir():
        p.error("task file or code directory does not exist")
    if args.start < 0 or args.timeout < 1 or args.workers < 1:
        p.error("start, timeout, and workers must be valid")
    return args

def main() -> None:
    args = arguments()
    started = time.perf_counter()
    selected = selected_tasks(load_tasks(args.task_file), args.start, args.count)
    if not selected:
        raise ValueError("No tasks selected")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    files = index_code(args.code_dir)
    ordered: list[dict | None] = [None] * len(selected)
    payloads: list[tuple] = []
    for pos, (source_index, task) in enumerate(selected):
        current_id = task_id(task, source_index)
        try:
            sample, total = prepare_task(task, args.limit_tests)
        except Exception as exc:
            ordered[pos] = failed(
                current_id, 0, "invalid_test_data", None, repr(exc)
            )
            continue
        if total == 0:
            ordered[pos] = failed(current_id, 0, "no_tests", None, "")
            continue
        path = files.get(current_id)
        if path is None:
            ordered[pos] = failed(current_id, total, "missing_code", None, "")
            continue
        payloads.append(
            (pos, current_id, sample, total, str(path), args.timeout, args.debug)
        )
    print(
        f"selected={len(selected)} scheduled={len(payloads)} "
        f"workers={args.workers} timeout_per_test={args.timeout}",
        flush=True,
    )
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
   



    completed = sum(
    record is not None
    for record in ordered
)

    with ctx.Pool(
        processes=args.workers,
        maxtasksperchild=1,
    ) as pool:
        for pos, record in pool.imap_unordered(
            evaluate,
            payloads,
            chunksize=1,
        ):
            workdir = record.pop("_workdir", None)

            if workdir:
                shutil.rmtree(
                    workdir,
                    ignore_errors=True,
                )

            ordered[pos] = record
            completed += 1

            if not args.quiet:
                status = (
                    "PASS"
                    if record["passed"]
                    else "FAIL"
                )
                print(
                    f"[{completed}/{len(selected)}] "
                    f"{status} {record['task_id']}: "
                    f"{record['failure_type']} "
                    f"(task_position={pos + 1})",
                    flush=True,
                )

    missing_results = [
        index
        for index, record in enumerate(ordered)
        if record is None
    ]

    if missing_results:
        raise RuntimeError(
            f"Evaluator returned no result for "
            f"{len(missing_results)} tasks: "
            f"{missing_results[:20]}"
        )

    details = list(ordered)
    
    
    
    
    passed = sum(r["passed"] for r in details)
    tests_run = sum(r["tests_run"] for r in details)
    tests_passed = sum(r["tests_passed"] for r in details)
    tests_total = sum(r["tests_total"] for r in details)
    failures = Counter(r["failure_type"] for r in details if not r["passed"])
    summary = {
        "stage": "taco_code_evaluation",
        "dataset": "taco",
        "workflow": args.workflow,
        "model": args.model_name,
        "task_file": str(args.task_file),
        "code_dir": str(args.code_dir),
        "results_dir": str(args.results_dir),
        "selected": len(selected),
        "evaluated": len(details),
        "passed": passed,
        "failed": len(selected) - passed,
        "pass_at_1": round(passed / len(selected), 6),
        "tasks_passing_at_least_one_test": sum(
            r["tests_passed"] > 0 for r in details
        ),
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_pass_rate": (
            round(tests_passed / tests_total, 6) if tests_total else 0.0
        ),
        "failure_counts": dict(sorted(failures.items())),
        "timeout_seconds_per_test": args.timeout,
        "workers": args.workers,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    save_json(args.results_dir / "details.json", details)
    save_json(args.results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()