from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.common.config import DATASETS, LOG_ROOT
from src.common.io_utils import load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.raw_contract_paths import raw_contract_path
from src.common.task_utils import select_tasks, task_identifier, task_prompt


STAGE = "stage_2a_livecodebench_contract_synthesis"
LOG_FILE = LOG_ROOT / "livecodebench_contract_synthesis.log"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_DELAY = 0.0


CONTRACT_SCHEMA = {
    "task_id": "",
    "benchmark": "livecodebench",
    "task_type": "stdin_stdout_program",
    "summary": "",
    "input_format": "",
    "output_format": "",
    "preconditions": [],
    "postconditions": [],
    "edge_cases": [],
    "complexity_notes": "",
}


SYSTEM_PROMPT = (
    "You synthesize a raw behavioral contract for a competitive-programming "
    "Python task. The task is a full stdin/stdout program, not a function. "
    "Return exactly one valid JSON object. No markdown. No explanation."
)


def build_contract_prompt(task: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Create a raw contract for this LiveCodeBench task.",
            "",
            "The solution must be a complete Python program that reads from stdin and writes to stdout.",
            "Do not describe a Python function signature unless the prompt explicitly provides one.",
            "",
            "Required JSON schema:",
            json.dumps(CONTRACT_SCHEMA, indent=2, ensure_ascii=False),
            "",
            "Field guidance:",
            "- summary: one sentence about the problem.",
            "- input_format: describe stdin structure.",
            "- output_format: describe stdout structure.",
            "- preconditions: valid input constraints from the prompt.",
            "- postconditions: observable guarantees about stdout.",
            "- edge_cases: boundary/special valid cases and expected behavior.",
            "- complexity_notes: any visible constraints that affect algorithm choice.",
            "",
            "Task ID:",
            str(task_identifier(task)),
            "",
            "Original prompt:",
            task_prompt(task),
        ]
    )


def parse_contract(raw_response: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_response)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed contract must be a JSON object.")

    parsed.setdefault("task_id", "")
    parsed.setdefault("benchmark", "livecodebench")
    parsed.setdefault("task_type", "stdin_stdout_program")
    parsed.setdefault("summary", "")
    parsed.setdefault("input_format", "")
    parsed.setdefault("output_format", "")
    parsed.setdefault("preconditions", [])
    parsed.setdefault("postconditions", [])
    parsed.setdefault("edge_cases", [])
    parsed.setdefault("complexity_notes", "")

    parsed["task_id"] = parsed["task_id"] or ""
    parsed["benchmark"] = "livecodebench"
    parsed["task_type"] = "stdin_stdout_program"

    return parsed


def build_record(
    *,
    task: dict[str, Any],
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
        "benchmark": "livecodebench",
        "task_type": "stdin_stdout_program",
        "entry_point": None,
        "stage": STAGE,
        "status": status,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "raw_response": raw_response,
        "api_result": api_result,
        "api_latency_seconds": (
            api_result.get("latency_seconds") if isinstance(api_result, dict) else None
        ),
        "error": error,
        "source": task.get("source"),
        "source_version": task.get("source_version"),
        "metadata": task.get("metadata"),
    }


def generate_one(
    *,
    task: dict[str, Any],
    provider: str,
    model: str,
    client: Any,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    raw_response = None
    api_result = None

    try:
        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_contract_prompt(task)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )

        contract = parse_contract(raw_response)

        return build_record(
            task=task,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            status="success",
            contract=contract,
            raw_response=raw_response,
            api_result=api_result,
        )

    except Exception as exc:
        logging.exception("LiveCodeBench contract synthesis failed for %s", task_identifier(task))

        return build_record(
            task=task,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            status="failed",
            raw_response=raw_response,
            api_result=api_result,
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiveCodeBench Stage 2A: synthesize stdin/stdout contracts"
    )

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
    args = parse_args()
    setup_logging(LOG_FILE)

    dataset = "livecodebench"
    info = DATASETS[dataset]
    provider = args.provider
    model = args.model or default_model(provider)
    client = get_client(provider)

    tasks = select_tasks(
        load_json_list(info["path"]),
        start=args.start,
        count=args.count,
    )

    done = failed = skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        task_id = task_identifier(task)
        out_file = raw_contract_path(dataset, task_id, provider, model)

        if out_file.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id}")
            continue

        record = generate_one(
            task=task,
            provider=provider,
            model=model,
            client=client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        save_json(out_file, record)

        if record["status"] == "success":
            done += 1
            print(f"[{index}] DONE {task_id}")
        else:
            failed += 1
            print(f"[{index}] FAIL {task_id}: {record['error']}")

        if args.delay > 0:
            time.sleep(args.delay)

    print("\nLiveCodeBench contract synthesis finished")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


if __name__ == "__main__":
    main()