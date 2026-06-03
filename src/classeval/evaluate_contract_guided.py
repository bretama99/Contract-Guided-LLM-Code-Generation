from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.classeval.core import JsonDict, task_id, text, verify_class_code
from src.classeval.execution import evaluate_generation, failure_record, generation_metadata, summarize_results
from src.classeval.generate_contracts import TASK_FILE
from src.classeval.generate_from_contracts import CONTRACT_SOURCES, generation_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model
from src.common.task_utils import select_tasks

RAW_STAGE = "2C_raw_contract_guided_evaluation"
OPT_STAGE = "2F_optimized_rl_contract_guided_evaluation"

RAW_EVAL_DIR = OUTPUT_ROOT / "classeval" / "evaluation" / "contract_guided"
OPT_EVAL_DIR = OUTPUT_ROOT / "classeval" / "evaluation" / "rl_optimized_contract_guided"

RAW_RESULT_DIR = RESULTS_ROOT / "classeval" / "contract_guided"
OPT_RESULT_DIR = RESULTS_ROOT / "classeval" / "rl_optimized_contract_guided"

LOG_FILE = LOG_ROOT / "classeval_contract_guided_eval.log"


def require_source(source: str) -> str:
    if source not in CONTRACT_SOURCES:
        raise ValueError(f"Unsupported contract source: {source}")
    return source


def stage_for(source: str) -> str:
    return RAW_STAGE if require_source(source) == "raw" else OPT_STAGE


def result_dir(source: str) -> Path:
    return RAW_RESULT_DIR if require_source(source) == "raw" else OPT_RESULT_DIR


def evaluation_path(task: JsonDict, provider: str, model: str, contract_source: str = "raw") -> Path:
    if require_source(contract_source) == "raw":
        from src.classeval.core import output_path

        return output_path(RAW_EVAL_DIR, task, provider, model, "eval")

    return OPT_EVAL_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def validation_failure(
    task: JsonDict,
    generation: JsonDict,
    gen_path: Path,
    *,
    provider: str,
    model: str,
    stage: str,
    error: Exception,
) -> JsonDict:
    record = failure_record(
        task,
        provider=provider,
        model=model,
        stage=stage,
        failure_type=str(error),
        error=str(error),
        generation_path=gen_path,
        failure_stage="generation_validation",
    )
    record["generation"] = generation_metadata(generation, gen_path)
    return record


def evaluate_one(
    task: JsonDict,
    provider: str,
    model: str,
    contract_source: str,
    timeout: float,
    include_code: bool,
) -> JsonDict:
    stage = stage_for(contract_source)
    gen_path = generation_path(task, provider, model, contract_source)

    if not gen_path.exists():
        return failure_record(
            task,
            provider=provider,
            model=model,
            stage=stage,
            failure_type="missing generation",
            error=f"Missing generation file: {gen_path}",
            generation_path=gen_path,
            failure_stage="generation",
        )

    generation = load_json(gen_path)

    if isinstance(generation, dict) and generation.get("status") == "success":
        try:
            verify_class_code(text(generation.get("generated_code") or generation.get("code")), task)
        except Exception as exc:
            return validation_failure(
                task,
                generation,
                gen_path,
                provider=provider,
                model=model,
                stage=stage,
                error=exc,
            )

    return evaluate_generation(
        task,
        generation,
        gen_path,
        provider=provider,
        model=model,
        stage=stage,
        timeout=timeout,
        include_code=include_code,
    )


def result_name(source: str, provider: str, model: str) -> str:
    label = "raw_contract_guided" if require_source(source) == "raw" else "optimized_rl_contract_guided"
    return f"classeval_{label}_{safe_name(provider)}_{safe_name(model)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClassEval raw or optimized contract-guided outputs")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-source", choices=CONTRACT_SOURCES, default="raw")
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
        path = evaluation_path(task, args.provider, model, args.contract_source)

        if path.exists() and not args.overwrite:
            result = load_json(path)
        else:
            result = evaluate_one(
                task,
                args.provider,
                model,
                args.contract_source,
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
        stage=stage_for(args.contract_source),
        timeout=args.timeout,
    )
    summary["contract_source"] = args.contract_source
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 4)

    name = result_name(args.contract_source, args.provider, model)
    out_dir = result_dir(args.contract_source)
    save_json(out_dir / f"{name}_summary.json", summary)
    save_json(out_dir / f"{name}_details.json", results)

    print(f"\nClassEval evaluation finished [{args.contract_source}].")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()