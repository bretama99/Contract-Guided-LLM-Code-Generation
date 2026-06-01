#!/usr/bin/env python3
from __future__ import annotations

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
    "Synthesize a raw, unvalidated Design-by-Contract specification for the Python benchmark task. "
    "Use only the prompt, signature, type hints, docstring, examples, imports, and visible helper code. "
    "Return exactly one valid JSON object. Do not generate code, tests, explanations, repairs, "
    "validation results, or markdown."
)

logger = logging.getLogger(__name__)


def parse_contract_response(raw_response: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed contract must be a JSON object.")
    return parsed


def make_record(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    status: str,
    contract: dict[str, Any] | None,
    raw_response: str | None,
    api_result: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "benchmark": benchmark,
        "dataset": dataset,
        "entry_point": task_entry_point(task),
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "raw_response": raw_response,
        "api_result": api_result,
        "api_latency_seconds": api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
        "error": error,
        "source": task.get("source"),
        "source_version": task.get("source_version"),
    }


def generate_one(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    client: Any,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    raw_response: str | None = None
    api_result: dict[str, Any] | None = None

    try:
        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_contract_prompt(task, dataset)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )

        parsed = parse_contract_response(raw_response)
        contract = normalize_contract(parsed, task, dataset)

        return make_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            status="success",
            contract=contract,
            raw_response=raw_response,
            api_result=api_result,
            error=None,
        )

    except Exception as exc:
        logger.exception("Raw contract synthesis failed for task %s", task_identifier(task))
        return make_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            status="failed",
            contract=None,
            raw_response=raw_response,
            api_result=api_result,
            error=str(exc),
        )


def run(args: argparse.Namespace) -> None:
    dataset_info = DATASETS[args.dataset]
    benchmark = dataset_info["label"]
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)

    tasks = select_tasks(
        load_json_list(dataset_info["path"]),
        start=args.start,
        count=args.count,
    )

    counts = {"completed": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        tid = task_identifier(task)
        out = raw_contract_path(args.dataset, tid, args.provider, model)

        if out.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}] SKIP {tid}")
            continue

        record = generate_one(
            task=task,
            dataset=args.dataset,
            benchmark=benchmark,
            provider=args.provider,
            model=model,
            client=client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        save_json(out, record)

        ok = record["status"] == "success"
        counts["completed" if ok else "failed"] += 1
        print(f"[{index}] {'DONE' if ok else 'FAIL'} {tid}" + ("" if ok else f": {record['error']}"))

        if args.delay > 0:
            time.sleep(args.delay)

    print("\nStage 2A raw contract synthesis finished")
    print(f"Dataset: {args.dataset} ({benchmark})")
    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Completed: {counts['completed']}")
    print(f"Failed: {counts['failed']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2A: synthesize raw unvalidated contracts")
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
    run(parse_args())


if __name__ == "__main__":
    main()