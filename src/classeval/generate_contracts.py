from __future__ import annotations

import argparse
import ast
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.classeval.contract_clause_utils import (
    contract_clause_errors,
    infer_clause_hints,
    merge_missing_clause_hints,
)
from src.classeval.contract_normalizer import normalize_contract_shape, stage2_shape_errors
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
DATASET = "classeval"
EXECUTION_MODEL = "class_level"
STAGE = "2A_raw_contract_synthesis"

TASK_FILE = ROOT / "data" / "processed" / "classeval" / "ClassEval_data.json"
PROMPT_FILE = ROOT / "prompts" / "classeval" / "contract_prompt.txt"
OUT_DIR = OUTPUT_ROOT / "classeval" / "contracts"
LOG_FILE = LOG_ROOT / "classeval_contracts.log"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_DELAY = 0.0
SYSTEM_PROMPT = (
    "You are an expert Design-by-Contract specification generator. "
    "Return exactly one valid JSON object. "
    "Do not include markdown, explanations, Python code, tests, reference solutions, or hidden answers."
)

logger = logging.getLogger(__name__)


def contract_path(task: JsonDict, provider: str, model: str):
    return output_path(OUT_DIR, task, provider, model, "contract")


def imports_for_prompt(task: JsonDict) -> list[str]:
    value = task.get("import_statement")
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [text(item) for item in value if text(item)]
    return []


def method_names(task: JsonDict) -> list[str]:
    names: list[str] = []

    for method in safe_methods_info(task):
        name = text(method.get("method_name"))
        if name and name not in names:
            names.append(name)

    try:
        profiles = method_profiles(skeleton(task), class_name(task))
    except SyntaxError:
        profiles = {}

    for name in profiles:
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
        "class_constructor": text(task.get("class_constructor")),
        "fields": as_list(task.get("fields")),
        "import_statement": imports_for_prompt(task),
        "method_names": method_names(task),
        "methods_info": safe_methods_info(task),
        "skeleton": skeleton(task),
    }


def build_prompt(task: JsonDict, template: str) -> str:
    task_view = sanitized_task(task)
    hints = infer_clause_hints(task_view)
    prompt = (
        template.replace("{schema}", compact_json(schema_for_prompt()))
        .replace("{structure}", compact_json(task_view))
    )
    prompt += "\n\nADDITIONAL_STAGE_2_CONTRACT_REQUIREMENTS:\n"
    prompt += "- Generate a faithful raw/v1 contract using only the visible task data.\n"
    prompt += "- Fill meaningful preconditions, postconditions, invariants, and edge_cases when supported by the visible task.\n"
    prompt += "- Do not leave every clause group empty for a method.\n"
    prompt += "- Preconditions describe valid inputs or required class state assumptions only.\n"
    prompt += "- Postconditions describe observable return values, state updates, side effects, or output format.\n"
    prompt += "- Invariants describe object state that remains valid before and after methods.\n"
    prompt += "- Edge cases describe visible boundary cases only when supported.\n"
    prompt += "- Do not invent exception behavior, hidden-test behavior, reference-solution logic, or exact test answers.\n"
    prompt += "\nVISIBLE_CLAUSE_HINTS:\n"
    prompt += compact_json(hints)
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


def class_node(task: JsonDict) -> ast.ClassDef | None:
    try:
        tree = ast.parse(skeleton(task))
    except SyntaxError:
        return None
    return next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name(task)),
        None,
    )


def function_header(source: str, node: ast.FunctionDef) -> str:
    segment = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    depth = 0
    quote: str | None = None
    escape = False

    for index, char in enumerate(segment):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0:
            return segment[:index].strip()

    return segment.splitlines()[0].strip()


def method_signature_map(task: JsonDict) -> dict[str, str]:
    source = skeleton(task)
    cls = class_node(task)
    if cls is None:
        return {}

    return {
        node.name: function_header(source, node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def method_static_map(task: JsonDict) -> dict[str, bool]:
    cls = class_node(task)
    if cls is None:
        return {}

    return {
        node.name: any(
            (isinstance(dec, ast.Name) and dec.id == "staticmethod")
            or (isinstance(dec, ast.Attribute) and dec.attr == "staticmethod")
            for dec in node.decorator_list
        )
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def repair_interface(contract: JsonDict, task: JsonDict) -> JsonDict:
    signatures = method_signature_map(task)
    static_flags = method_static_map(task)
    expected_methods = method_names(task)

    contract["class_interface"] = {
        "fields": as_list(task.get("fields")),
        "methods": expected_methods,
    }

    constructor = as_dict(contract.get("constructor")).copy()
    constructor["signature"] = signatures.get("__init__", text(constructor.get("signature")))
    constructor.setdefault("contract", {})
    contract["constructor"] = constructor

    existing = {
        text(method.get("method_name")): method
        for method in as_list(contract.get("method_contracts"))
        if isinstance(method, dict) and text(method.get("method_name"))
    }

    repaired_methods: list[JsonDict] = []
    for name in expected_methods:
        method = as_dict(existing.get(name)).copy()
        method["method_name"] = name
        method["signature"] = signatures.get(name, text(method.get("signature")))
        method["is_static"] = bool(static_flags.get(name, method.get("is_static") is True))
        method.setdefault("dependencies", {})
        method.setdefault("contract", {})
        repaired_methods.append(method)

    contract["method_contracts"] = repaired_methods
    return contract


def metadata_contract(task: JsonDict) -> JsonDict:
    return repair_interface(stamp_contract(new_contract_schema(), task), task)


def parse_raw_response(response: str, task: JsonDict) -> tuple[JsonDict, str | None]:
    try:
        raw = extract_json_object(response)
        if isinstance(raw, dict):
            return raw, None
    except Exception as exc:
        logger.exception("Could not parse LLM contract JSON for %s", task_id(task))
        return metadata_contract(task), str(exc)

    return metadata_contract(task), "LLM response did not contain a JSON object."


def finalize_contract(raw: JsonDict, task: JsonDict) -> tuple[JsonDict, list[str], list[str]]:
    task_view = sanitized_task(task)

    contract = normalize_contract_shape(stamp_contract(raw, task))
    contract = repair_interface(contract, task)
    contract = merge_missing_clause_hints(contract, task_view)
    contract = normalize_contract_shape(contract)
    contract = repair_interface(contract, task)

    shape_errors = stage2_shape_errors(contract, expected_methods=method_names(task))
    clause_warnings = contract_clause_errors(contract, task_view)
    return contract, shape_errors, clause_warnings


def parse_contract(response: str, task: JsonDict) -> tuple[JsonDict, list[str], list[str], str | None]:
    raw, parse_error = parse_raw_response(response, task)
    contract, shape_errors, clause_warnings = finalize_contract(raw, task)
    return contract, shape_errors, clause_warnings, parse_error


def make_record(task: JsonDict, args: argparse.Namespace, model: str, **extra: Any) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": BENCHMARK,
        "dataset": DATASET,
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


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any, template: str) -> JsonDict:
    prompt = ""
    response = ""
    api_result: JsonDict | None = None
    shape_errors: list[str] = []
    clause_warnings: list[str] = []
    parse_error: str | None = None

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

        contract, shape_errors, clause_warnings, parse_error = parse_contract(response, task)

        if shape_errors:
            raise ValueError("Stage 2 shape errors: " + "; ".join(shape_errors[:10]))

        return make_record(
            task,
            args,
            model,
            status="success",
            contract=contract,
            contract_type="baseline",
            contract_version="v1_baseline_contract",
            contract_generation_mode="metadata_fallback" if parse_error else "llm_json_plus_visible_hints",
            llm_parse_error=parse_error,
            stage2_shape_errors=[],
            stage2_clause_warnings=clause_warnings,
            generation_prompt=prompt,
            raw_response=response,
            api_result=api_result,
            api_latency_seconds=api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
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
            contract_type="baseline",
            contract_version="v1_baseline_contract",
            contract_generation_mode="failed",
            llm_parse_error=parse_error,
            stage2_shape_errors=shape_errors,
            stage2_clause_warnings=clause_warnings,
            generation_prompt=prompt,
            raw_response=response,
            api_result=api_result,
            api_latency_seconds=api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval Stage 2 raw contracts")
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

    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    template = read_template(PROMPT_FILE, ("schema", "structure"))
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = failed = skipped = fallback = 0
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
            fallback += int(record.get("contract_generation_mode") == "metadata_fallback")
            warnings = len(record.get("stage2_clause_warnings") or [])
            suffix = " fallback" if record.get("contract_generation_mode") == "metadata_fallback" else ""
            print(f"[{index}] DONE {task_id(task)} warnings={warnings}{suffix}")
        else:
            failed += 1
            print(f"[{index}] FAILED {task_id(task)}: {record['error']}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval Stage 2 raw contract synthesis finished.")
    print(f"Completed: {done}")
    print(f"Metadata fallback: {fallback}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


if __name__ == "__main__":
    main()