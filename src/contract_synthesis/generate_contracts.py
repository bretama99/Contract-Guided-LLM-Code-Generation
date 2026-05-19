#!/usr/bin/env python3
import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any
from src.common.config import DATASETS, LOG_ROOT
from src.common.io_utils import load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.raw_contract_paths import raw_contract_path
from src.common.task_utils import select_tasks, task_entry_point, task_identifier
from src.contract_synthesis.contract_normalizer import normalize_contract
from src.contract_synthesis.contract_schema import build_contract_prompt
from src.contract_synthesis.contract_schema_constants import SCHEMA_VERSION

STAGE = "stage_2a_raw_contract_synthesis"
LOG_FILE = LOG_ROOT / "stage2_contract_synthesis.log"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 3000
DEFAULT_DELAY = 0.0
SYSTEM_PROMPT = (
    "Synthesize a raw, unvalidated contract for the Python benchmark task. "
    "Use only the prompt, signature, type hints, docstring, and examples as evidence. "
    "Return exactly one valid JSON object. Do not generate code, tests, explanations, "
    "repairs, validation results, or markdown."
)

def parse_contract_response(raw_response: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed contract must be a JSON object.")
    return parsed

def build_record(
    *,
    task: dict[str, Any],
    benchmark: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    status: str,
    contract: dict[str, Any] | None = None,
    raw_response: str | None = None,
    api_result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "benchmark": benchmark,
        "entry_point": task_entry_point(task),
        "stage": STAGE,
        "status": status,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "contract": contract,
        "raw_response": raw_response,
        "api_result": api_result,
        "api_latency_seconds": (
            api_result.get("latency_seconds")
            if isinstance(api_result, dict)
            else None
        ),
        "error": error,
        "source": task.get("source"),
        "source_version": task.get("source_version"),
    }

def generate_contract_for_task(
    *,
    task: dict[str, Any],
    benchmark: str,
    provider: str,
    model: str,
    client: Any,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    raw_response: str | None = None
    api_result: dict[str, Any] | None = None
    common = {
        "task": task,
        "benchmark": benchmark,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_contract_prompt(task, benchmark)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        parsed_contract = parse_contract_response(raw_response)
        normalized_contract = normalize_contract(parsed_contract, task, benchmark)
        return build_record(
            **common,
            status="success",
            contract=normalized_contract,
            raw_response=raw_response,
            api_result=api_result,
        )

    except Exception as exc:
        logging.exception(
            "Raw contract synthesis failed for task %s",
            task_identifier(task),
        )
        return build_record(
            **common,
            status="failed",
            raw_response=raw_response,
            api_result=api_result,
            error=str(exc),
        )

def generate_contracts(args: argparse.Namespace) -> None:
    dataset_config = DATASETS[args.dataset]
    benchmark = args.dataset
    benchmark_label = dataset_config["label"]
    provider = args.provider
    model = args.model or default_model(provider)
    client = get_client(provider)
    tasks = select_tasks(
        load_json_list(dataset_config["path"]),
        start=args.start,
        count=args.count,
    )
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    started_at = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        task_id = task_identifier(task)
        output_path = raw_contract_path(args.dataset, task_id, provider, model)
        if output_path.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}] SKIP {task_id}")
            continue
        record = generate_contract_for_task(
            task=task,
            benchmark=benchmark,
            provider=provider,
            model=model,
            client=client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        save_json(output_path, record)

        if record["status"] == "success":
            counts["completed"] += 1
            print(f"[{index}] DONE {task_id}")
        else:
            counts["failed"] += 1
            print(f"[{index}] FAIL {task_id}: {record['error']}")
        if args.delay > 0:
            time.sleep(args.delay)
    elapsed = round(time.perf_counter() - started_at, 4)
    print("\nStage 2A raw contract synthesis finished")
    print(f"Dataset: {args.dataset} ({benchmark_label})")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Completed: {counts['completed']}")
    print(f"Failed: {counts['failed']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Elapsed: {elapsed}s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2A: synthesize raw unvalidated contracts"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    setup_logging(LOG_FILE)
    generate_contracts(parse_args())

if __name__ == "__main__":
    main()