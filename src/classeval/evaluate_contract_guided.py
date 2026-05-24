from __future__ import annotations
import argparse
import json
import time
from src.classeval.core import JsonDict, output_path, task_id
from src.classeval.execution import (
    evaluate_generation,
    failure_record,
    summarize_results,
)
from src.common.config import LOG_ROOT, OUTPUT_ROOT, RESULTS_ROOT, ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model
from src.common.task_utils import select_tasks

TASK_FILE = ROOT / "data" / "processed" / "classeval" / "ClassEval_data.json"
GEN_DIR = OUTPUT_ROOT / "classeval" / "contract_guided"
EVAL_DIR = OUTPUT_ROOT / "classeval" / "evaluation" / "contract_guided"
RESULT_DIR = RESULTS_ROOT / "classeval" / "contract_guided"
LOG_FILE = LOG_ROOT / "classeval_contract_guided_eval.log"
STAGE = "2C_raw_contract_guided_evaluation"

def generation_path(task: JsonDict, provider: str, model: str):
    return output_path(GEN_DIR, task, provider, model, "contract_guided")

def evaluation_path(task: JsonDict, provider: str, model: str):
    return output_path(EVAL_DIR, task, provider, model, "eval")

def evaluate_one(
    task: JsonDict,
    provider: str,
    model: str,
    timeout: float,
    include_code: bool,
) -> JsonDict:
    path = generation_path(task, provider, model)
    if not path.exists():
        return failure_record(
            task,
            provider=provider,
            model=model,
            stage=STAGE,
            failure_type="missing_generation",
            error=f"Missing generation file: {path}",
            generation_path=path,
        )
    return evaluate_generation(
        task,
        load_json(path),
        path,
        provider=provider,
        model=model,
        stage=STAGE,
        timeout=timeout,
        include_code=include_code,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClassEval contract-guided outputs")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)
    model = args.model or default_model(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)
    results: list[JsonDict] = []
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = evaluation_path(task, args.provider, model)
        if path.exists() and not args.overwrite:
            result = load_json(path)
        else:
            result = evaluate_one(
                task,
                args.provider,
                model,
                args.timeout,
                args.include_code,
            )
            save_json(path, result)

        result["index"] = index
        results.append(result)
        status = "PASS" if result.get("passed") else f"FAIL/{result.get('failure_type')}"
        print(f"[{index}] {status}: {task_id(task)}")

    summary = summarize_results(
        results,
        provider=args.provider,
        model=model,
        stage=STAGE,
        timeout=args.timeout,
    )
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    name = f"classeval_contract_guided_{safe_name(args.provider)}_{safe_name(model)}"
    save_json(RESULT_DIR / f"{name}_summary.json", summary)
    save_json(RESULT_DIR / f"{name}_details.json", results)

    print("\nClassEval contract-guided evaluation finished.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()