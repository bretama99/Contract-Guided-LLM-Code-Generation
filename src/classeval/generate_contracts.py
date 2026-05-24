from __future__ import annotations
import argparse
import logging
import ast
import time
from datetime import datetime, timezone
from typing import Any
from src.classeval.contract_normalizer import (
    normalize_contract_shape,
    stage2_shape_errors,
)
from src.classeval.contract_schema import new_contract_schema, schema_for_prompt
from src.classeval.core import (
    JsonDict,
    as_dict,
    as_list,
    class_name,
    compact_json,
    ensure_no_reference_leak,
    method_profiles,
    output_path,
    read_template,
    safe_methods_info,
    skeleton,
    task_id,
    text,
)
from src.common.config import LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.task_utils import select_tasks

BENCHMARK = "ClassEval"
EXECUTION_MODEL = "class_level"
STAGE = "2A_raw_contract_synthesis"
TASK_FILE = ROOT / "data" / "processed" / "classeval" / "ClassEval_data.json"
PROMPT_FILE = ROOT / "prompts" / "classeval" / "contract_prompt.txt"
OUT_DIR = OUTPUT_ROOT / "classeval" / "contracts"
LOG_FILE = LOG_ROOT / "classeval_contracts.log"
DEFAULT_MAX_TOKENS = 4096
SYSTEM_PROMPT = (
    "You are an expert Design-by-Contract specification generator. "
    "Return exactly one valid JSON object. "
    "Do not include markdown, explanations, Python code, or tests."
)
logger = logging.getLogger(__name__)

def contract_path(task: JsonDict, provider: str, model: str):
    return output_path(OUT_DIR, task, provider, model, "contract")

def method_names(task: JsonDict) -> list[str]:
    names: list[str] = []
    for method in safe_methods_info(task):
        name = text(method.get("method_name"))
        if name and name not in names:
            names.append(name)
    for name in method_profiles(skeleton(task), class_name(task)):
        if name != "__init__" and name not in names:
            names.append(name)
    return names

def sanitized_task(task: JsonDict) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": BENCHMARK,
        "execution_model": EXECUTION_MODEL,
        "class_name": class_name(task),
        "class_description": text(task.get("class_description")),
        "class_constructor": task.get("class_constructor") or "",
        "fields": as_list(task.get("fields")),
        "import_statement": as_list(task.get("import_statement")),
        "method_names": method_names(task),
        "methods_info": safe_methods_info(task),
        "skeleton": skeleton(task),
    }

def build_prompt(task: JsonDict, template: str) -> str:
    prompt = (
        template.replace("{schema}", compact_json(schema_for_prompt()))
        .replace("{structure}", compact_json(sanitized_task(task)))
    )
    ensure_no_reference_leak(task, prompt)
    return prompt

def stamp_contract(raw: JsonDict, task: JsonDict) -> JsonDict:
    contract = new_contract_schema()
    contract.update(raw)
    task_block = as_dict(contract.get("task"))
    task_block.update(
        {
            "task_id": task_id(task),
            "benchmark": BENCHMARK,
            "language": "python",
            "execution_model": EXECUTION_MODEL,
            "class_name": class_name(task),
            "entry_point": class_name(task),
        }
    )
    if not text(task_block.get("summary")):
        task_block["summary"] = text(task.get("class_description"))
    contract["task"] = task_block
    return contract

def parse_contract(response: str, task: JsonDict) -> tuple[JsonDict, list[str]]:
    raw = extract_json_object(response)

    if not isinstance(raw, dict):
        raise ValueError("LLM response does not contain a JSON object.")

    contract = normalize_contract_shape(stamp_contract(raw, task))
    contract = repair_method_signatures(contract, task)

    errors = stage2_shape_errors(contract, expected_methods=method_names(task))
    return contract, errors

def make_record(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    **extra: Any,
) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": BENCHMARK,
        "execution_model": EXECUTION_MODEL,
        "class_name": class_name(task),
        "entry_point": class_name(task),
        "stage": STAGE,
        "status": extra.pop("status"),
        "provider": args.provider,
        "model_name": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "ClassEval_data.json",
        **extra,
    }

def generate_one(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    client: Any,
    template: str,
) -> JsonDict:
    response = ""
    api_result: JsonDict | None = None
    errors: list[str] = []
    try:
        prompt = build_prompt(task, template)
        response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=True,
        )
        contract, errors = parse_contract(response, task)
        if errors:
            raise ValueError("Stage 2 shape errors: " + "; ".join(errors[:10]))
        return make_record(
            task,
            args,
            model,
            status="success",
            contract=contract,
            stage2_shape_errors=[],
            raw_response=response,
            api_result=api_result,
            error=None,
        )
    except Exception as exc:
        logger.exception("Contract synthesis failed for %s", task_id(task))
        return make_record(
            task,
            args,
            model,
            status="failed",
            contract=None,
            stage2_shape_errors=errors,
            raw_response=response,
            api_result=api_result,
            error=str(exc),
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval Stage 2 raw contracts")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
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
    template = read_template(PROMPT_FILE, ("schema", "structure"))
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)
    done = failed = skipped = 0
    started = time.perf_counter()
    
    for index, task in enumerate(tasks, start=args.start):
        path = contract_path(task, args.provider, model)
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
    print("\nClassEval Stage 2 raw contract synthesis finished.")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")
def method_signature_map(task: JsonDict) -> dict[str, str]:
    tree = ast.parse(skeleton(task))
    cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name(task)
        ),
        None,
    )

    if cls is None:
        return {}

    signatures: dict[str, str] = {}

    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            source = ast.get_source_segment(skeleton(task), node) or ""
            header = source.split(":", 1)[0].strip()
            signatures[node.name] = header

    return signatures


def repair_method_signatures(contract: JsonDict, task: JsonDict) -> JsonDict:
    signatures = method_signature_map(task)

    for method in contract.get("method_contracts", []):
        if not isinstance(method, dict):
            continue

        name = text(method.get("method_name"))
        if name and not text(method.get("signature")) and name in signatures:
            method["signature"] = signatures[name]

    return contract

if __name__ == "__main__":
    main()