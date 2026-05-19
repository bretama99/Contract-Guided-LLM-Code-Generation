from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.common.config import DATASETS, LOG_ROOT
from src.common.io_utils import load_json, load_json_list, save_json, setup_logging
from src.common.llm_clients import call_chat_model, default_model, get_client
from src.common.parsing import extract_python_code
from src.common.raw_contract_paths import raw_contract_code_path, raw_contract_path
from src.common.task_utils import select_tasks, task_identifier, task_prompt


STAGE = "stage_2b_livecodebench_contract_guided_generation"

SYSTEM_PROMPT = (
    "Generate a complete executable Python solution for a competitive-programming "
    "stdin/stdout task. Return only Python source code. No markdown. No explanation."
)


def extract_contract(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Raw contract file must contain a JSON object.")

    if record.get("status") not in (None, "success"):
        raise RuntimeError("Raw contract status is not success.")

    contract = record.get("contract", record)

    if not isinstance(contract, dict):
        raise ValueError("Raw contract record does not contain a contract object.")

    return contract


def compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": contract.get("summary", ""),
        "input_format": contract.get("input_format", ""),
        "output_format": contract.get("output_format", ""),
        "preconditions": contract.get("preconditions", []),
        "postconditions": contract.get("postconditions", []),
        "edge_cases": contract.get("edge_cases", []),
        "complexity_notes": contract.get("complexity_notes", ""),
    }


def build_prompt(task: dict[str, Any], contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are solving a LiveCodeBench competitive-programming task.",
            "",
            "OUTPUT RULES:",
            "- Return only Python source code.",
            "- Do not include markdown or explanations.",
            "- Read all input from stdin.",
            "- Write only the required answers to stdout.",
            "- Prefer a solve() function called under if __name__ == '__main__'.",
            "- Do not print debug text.",
            "",
            "The raw contract is only a helper checklist.",
            "If the contract conflicts with the original prompt, follow the original prompt.",
            "",
            "ORIGINAL PROMPT:",
            task_prompt(task),
            "",
            "RAW CONTRACT:",
            json.dumps(compact_contract(contract), indent=2, ensure_ascii=False),
        ]
    )


def make_record(
    task: dict[str, Any],
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "benchmark": "livecodebench",
        "task_type": "stdin_stdout_program",
        "entry_point": None,
        "stage": STAGE,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": task.get("source"),
        "source_version": task.get("source_version"),
        "metadata": task.get("metadata"),
        **extra,
    }


def generate_one(
    task: dict[str, Any],
    args: argparse.Namespace,
    model: str,
    client: Any,
) -> dict[str, Any]:
    dataset = "livecodebench"
    task_id = task_identifier(task)
    contract_file = raw_contract_path(dataset, task_id, args.provider, model)

    prompt = ""
    raw_response = ""
    api_result = None
    code = None

    try:
        if not contract_file.exists():
            raise FileNotFoundError(f"Missing raw contract file: {contract_file}")

        contract = extract_contract(load_json(contract_file))
        prompt = build_prompt(task, contract)

        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=False,
        )

        code = extract_python_code(
            raw_response,
            entry_point=None,
            validate=True,
        )

        return make_record(
            task,
            args.provider,
            model,
            args.temperature,
            args.max_tokens,
            status="success",
            contract_path=str(contract_file),
            generation_prompt=prompt,
            raw_response=raw_response,
            generated_code=code,
            api_result=api_result,
            error=None,
        )

    except Exception as exc:
        logging.exception("LiveCodeBench contract-guided generation failed for %s", task_id)

        return make_record(
            task,
            args.provider,
            model,
            args.temperature,
            args.max_tokens,
            status="failed",
            contract_path=str(contract_file),
            generation_prompt=prompt,
            raw_response=raw_response,
            generated_code=code,
            api_result=api_result,
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiveCodeBench Stage 2B: generate code from raw contracts"
    )

    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_ROOT / "livecodebench_contract_guided_generation.log")

    dataset = "livecodebench"
    info = DATASETS[dataset]
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)

    tasks = select_tasks(
        load_json_list(info["path"]),
        start=args.start,
        count=args.count,
    )

    done = failed = skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        task_id = task_identifier(task)
        out_file = raw_contract_code_path(dataset, task_id, args.provider, model)

        if out_file.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id}")
            continue

        record = generate_one(task, args, model, client)
        save_json(out_file, record)

        if record["status"] == "success":
            done += 1
            print(f"[{index}] DONE {task_id}")
        else:
            failed += 1
            print(f"[{index}] FAIL {task_id}: {record['error']}")

        if args.delay > 0:
            time.sleep(args.delay)

    print("\nLiveCodeBench contract-guided generation finished")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


if __name__ == "__main__":
    main()