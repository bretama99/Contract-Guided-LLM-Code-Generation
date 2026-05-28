from __future__ import annotations
import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any
from src.classeval.core import (
    JsonDict,
    as_dict,
    as_list,
    class_name,
    ensure_no_reference_leak,
    extract_class_code,
    fill_template,
    output_path,
    pretty_json,
    read_template,
    skeleton,
    task_id,
    text,
)
from src.common.config import LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json, load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.task_utils import select_tasks

BENCHMARK = "ClassEval"
EXECUTION_MODEL = "class_level"
STAGE = "2B_raw_contract_conditioned_generation"
TASK_FILE = ROOT / "data" / "processed" / "classeval" / "ClassEval_data.json"
PROMPT_FILE = ROOT / "prompts" / "classeval" / "contract_guided_prompt.txt"
CONTRACT_DIR = OUTPUT_ROOT / "classeval" / "contracts"
OUT_DIR = OUTPUT_ROOT / "classeval" / "contract_guided"
LOG_FILE = LOG_ROOT / "classeval_contract_guided.log"
DEFAULT_MAX_TOKENS = 8192
SYSTEM_PROMPT = (
    "You are an expert Python code generation assistant. "
    "Return only complete, syntactically valid Python source code. "
    "Do not include markdown, code fences, explanations, or tests."
)
logger = logging.getLogger(__name__)

def generation_path(task: JsonDict, provider: str, model: str):
    return output_path(OUT_DIR, task, provider, model, "contract_guided")

def contract_path(task: JsonDict, provider: str, model: str):
    return output_path(CONTRACT_DIR, task, provider, model, "contract")

def prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := prune(item)) not in ("", None, [], {})
        }

    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := prune(item)) not in ("", None, [], {})
        ]
    return value

def compact_clause_items(items: Any) -> list[Any]:
    compacted = []

    for item in as_list(items):
        if not isinstance(item, dict):
            if text(item):
                compacted.append(item)
            continue

        compacted.append(
            prune(
                {
                    "description": item.get("description"),
                    "condition": item.get("condition"),
                    "expected_behavior": item.get("expected_behavior"),
                    "kind": item.get("kind"),
                    "source": item.get("source"),
                }
            )
        )

    return compacted


def compact_callable_contract(contract: JsonDict) -> JsonDict:
    invalid = as_dict(contract.get("invalid_input_behavior"))

    return prune(
        {
            "inputs": as_dict(contract.get("interface")).get("inputs"),
            "output": as_dict(contract.get("interface")).get("output"),
            "preconditions": compact_clause_items(contract.get("preconditions")),
            "postconditions": compact_clause_items(contract.get("postconditions")),
            "invariants": compact_clause_items(contract.get("invariants")),
            "edge_cases": compact_clause_items(contract.get("edge_cases")),
            "invalid_input_behavior": invalid if invalid.get("specified") is True else {},
        }
    )


def compact_method(method: JsonDict) -> JsonDict:
    return prune(
        {
            "method_name": method.get("method_name"),
            "signature": method.get("signature"),
            "dependencies": method.get("dependencies"),
            "contract": compact_callable_contract(as_dict(method.get("contract"))),
        }
    )


def compact_contract(contract: JsonDict) -> JsonDict:
    task = as_dict(contract.get("task"))
    constructor = as_dict(contract.get("constructor"))

    return prune(
        {
            "summary": task.get("summary"),
            "constructor": {
                "signature": constructor.get("signature"),
                "initializes": constructor.get("initializes"),
                "contract": compact_callable_contract(as_dict(constructor.get("contract"))),
            },
            "class_invariants": compact_clause_items(contract.get("class_invariants")),
            "method_contracts": [
                compact_method(method)
                for method in as_list(contract.get("method_contracts"))
                if isinstance(method, dict)
            ],
            "interaction_contracts": [
                prune(
                    {
                        "name": item.get("name"),
                        "method_sequence": item.get("method_sequence"),
                        "preconditions": compact_clause_items(item.get("preconditions")),
                        "postconditions": compact_clause_items(item.get("postconditions")),
                    }
                )
                for item in as_list(contract.get("interaction_contracts"))
                if isinstance(item, dict)
            ],
        }
    )
    
def load_contract(task: JsonDict, provider: str, model: str) -> JsonDict:
    path = contract_path(task, provider, model)
    if not path.exists():
        raise FileNotFoundError(f"Missing contract file: {path}")
    record = load_json(path)
    if record.get("status") != "success":
        raise RuntimeError(text(record.get("error")) or f"Contract generation failed: {path}")
    contract = record.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Contract record has no contract object: {path}")
    return compact_contract(contract)

def build_prompt(task: JsonDict, contract: JsonDict, template: str) -> str:
    prompt = fill_template(
        template,
        {
            "class_name": class_name(task),
            "skeleton": skeleton(task),
            "contract": pretty_json(contract),
        },
    )
    ensure_no_reference_leak(task, prompt)
    return prompt

def make_record(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    *,
    status: str,
    contract: JsonDict | None,
    prompt: str,
    raw_response: str,
    generated_code: str | None,
    api_result: JsonDict | None,
    error: str | None,
) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": BENCHMARK,
        "execution_model": EXECUTION_MODEL,
        "class_name": class_name(task),
        "entry_point": class_name(task),
        "stage": STAGE,
        "status": status,
        "provider": args.provider,
        "model_name": model,
        "contract_provider": args.contract_provider or args.provider,
        "contract_model": args.contract_model or model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "generation_prompt": prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "api_result": api_result,
        "error": error,
        "source": "ClassEval_data.json",
    }
def generate_one(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    client: Any,
    template: str,
) -> JsonDict:
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model
    contract: JsonDict | None = None
    prompt = ""
    raw_response = ""
    api_result: JsonDict | None = None

    try:
        contract = load_contract(task, contract_provider, contract_model)
        prompt = build_prompt(task, contract, template)

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

        return make_record(
            task,
            args,
            model,
            status="success",
            contract=contract,
            prompt=prompt,
            raw_response=raw_response,
            generated_code=extract_class_code(raw_response, task),
            api_result=api_result,
            error=None,
        )

    except Exception as exc:
        logger.exception("Contract-guided generation failed for %s", task_id(task))
        return make_record(
            task,
            args,
            model,
            status="failed",
            contract=contract,
            prompt=prompt,
            raw_response=raw_response,
            generated_code=None,
            api_result=api_result,
            error=str(exc),
        )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ClassEval code from skeleton and raw Stage-2 contract."
    )
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-provider")
    parser.add_argument("--contract-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    template = read_template(PROMPT_FILE, ("class_name", "skeleton", "contract"))
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)
    done = failed = skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = generation_path(task, args.provider, model)
        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        record = generate_one(task, args, model, client, template)
        save_json(path, record)
        if record["status"] == "success":
            done += 1
            print(f"[{index}] DONE {task_id(task)}")
        else:
            failed += 1
            print(f"[{index}] FAILED {task_id(task)}: {record['error']}")
        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval contract-guided generation finished.")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")

if __name__ == "__main__":
    main()