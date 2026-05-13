import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any

from huggingface_hub import dataset_info
from src.common.config import DATASETS, LOG_ROOT
from src.common.io_utils import load_json_list, save_json, setup_logging
from src.common.llm_clients import call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.raw_contract_paths import raw_contract_path
from src.common.task_utils import select_tasks, task_identifier, task_entry_point
from src.contract_synthesis.contract_normalizer import normalize_contract
from src.contract_synthesis.contract_schema import build_contract_prompt
from src.contract_synthesis.contract_schema_constants import SCHEMA_VERSION

STAGE = "stage_2a_raw_contract_synthesis"
LOG_FILE = LOG_ROOT / "stage2_contract_synthesis.log"
SYSTEM_PROMPT = (
    "Synthesize a raw, unvalidated contract for the Python benchmark task. "
    "Use only the prompt, signature, type hints, docstring, and examples as evidence. "
    "Return exactly one valid JSON object. Do not generate code, tests, explanations, "
    "repairs, validation results, or markdown."
)
RECOVERY_PROMPT = (
    "Recover malformed JSON formatting. "
    "Return exactly one valid JSON object only."
)

def recover_json_response(
    *,
    client: Any,
    model: str,
    broken_json: str,
    max_tokens: int,
) -> str:
    prompt = "\n".join(
        (
            "Recover the following malformed JSON text into exactly one valid JSON object.",
            "Preserve the same structure and meaning.",
            "Do not add or remove contract clauses.",
            "Return JSON only.",
            "",
            "Malformed JSON:",
            broken_json,
        )
    )
    response, _ = call_chat_model(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": RECOVERY_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        json_mode=True,
    )
    return response

def parse_contract_response(
    *,
    client: Any,
    model: str,
    raw_response: str,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        parsed = extract_json_object(raw_response)
    except Exception:
        repaired = recover_json_response(
            client=client,
            model=model,
            broken_json=raw_response,
            max_tokens=max_tokens,
        )
        parsed = extract_json_object(repaired)
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
        "speed": api_result,
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
        parsed = parse_contract_response(
            client=client,
            model=model,
            raw_response=raw_response,
            max_tokens=max_tokens,
        )
        contract = normalize_contract(parsed, task, benchmark)
        return build_record(
            **common,
            status="success",
            contract=contract,
            raw_response=raw_response,
            api_result=api_result,
        )
    except Exception as exc:
        logging.exception("Raw contract synthesis failed for task %s", task_identifier(task))
        return build_record(
            **common,
            status="failed",
            raw_response=raw_response,
            api_result=api_result,
            error=str(exc),
        )

def generate_contracts(args: argparse.Namespace) -> None:
    
    dataset_info = DATASETS[args.dataset]
    benchmark = args.dataset
    benchmark_label = dataset_info["label"]
    provider = args.provider
    model = args.model or default_model(provider)
    client = get_client(provider)
    tasks = select_tasks(
        load_json_list(dataset_info["path"]),
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
        success = record["status"] == "success"
        counts["completed" if success else "failed"] += 1
        if success:
            print(f"[{index}] DONE {task_id}")
        else:
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
        description="Synthesize raw unvalidated contracts"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    setup_logging(LOG_FILE)
    generate_contracts(parse_args())

if __name__ == "__main__":
    main()