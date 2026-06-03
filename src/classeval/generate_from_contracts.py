from __future__ import annotations

import argparse
import ast
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.core import (
    JsonDict,
    as_dict,
    as_list,
    class_name,
    ensure_no_reference_leak,
    extract_class_code,
    output_path,
    pretty_json,
    safe_methods_info,
    skeleton,
    task_id,
    text,
)
from src.classeval.generate_contracts import TASK_FILE
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.task_utils import select_tasks

BENCHMARK = "ClassEval"
DATASET = "classeval"
EXECUTION_MODEL = "class_level"

CONTRACT_SOURCES = ("raw", "optimized_rl")

RAW_STAGE = "2B_raw_contract_conditioned_generation"
OPT_STAGE = "2E_optimized_rl_contract_conditioned_generation"

CONTRACT_DIR = OUTPUT_ROOT / "classeval" / "contracts"
OPT_CONTRACT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contracts"

RAW_OUT_DIR = OUTPUT_ROOT / "classeval" / "contract_guided"
OPT_OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contract_guided"

LOG_FILE = LOG_ROOT / "classeval_contract_guided.log"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_DELAY = 0.0

SYSTEM_PROMPT = (
    "You generate complete Python class implementations. "
    "Return only Python source code. No markdown, explanations, tests, or partial code."
)

PROMPT = """
Generate one complete Python class for ClassEval.

Stage:
- This is Stage 2 contract-guided generation only.
- Make one direct implementation from the visible task and contract.
- Do not repair, self-correct, use test feedback, or infer hidden tests.

Hard rules:
- Implement the class from the required interface below.
- Preserve imports, class name/header, constructor signature, method signatures, return annotations, and decorators exactly.
- The visible task descriptions, method descriptions, parameter/return descriptions, and visible examples define expected behavior.
- The contract is guidance only; it must not override the visible task or required interface.
- Implement every required non-constructor method with executable logic.
- A constructor may be empty only when no visible state initialization is required.
- Do not output pass, ellipsis, TODO, NotImplementedError, empty non-constructor bodies, tests, markdown, or explanations.
- Do not use reference solutions, hidden tests, canonical answers, or test-specific leakage.
- Preserve return container types exactly: list vs tuple, dict vs list, bool vs None, printed output vs returned output.
- Return only Python source code.

Required interface:
{interface}

Visible task context:
{task_context}

Contract source:
{contract_source}

Contract metadata:
{metadata}

Selected contract:
{contract}
""".strip()

logger = logging.getLogger(__name__)


def require_source(source: str) -> str:
    if source not in CONTRACT_SOURCES:
        raise ValueError(f"Unsupported contract source: {source}")
    return source


def stage_for(source: str) -> str:
    return RAW_STAGE if require_source(source) == "raw" else OPT_STAGE


def contract_type(source: str) -> str:
    return "baseline" if require_source(source) == "raw" else "rl_optimized"


def contract_version(source: str) -> str:
    return "v1_baseline_contract" if require_source(source) == "raw" else "v2_rl_optimized"


def raw_contract_path(task: JsonDict, provider: str, model: str) -> Path:
    return output_path(CONTRACT_DIR, task, provider, model, "contract")


def optimized_contract_path(task: JsonDict, provider: str, model: str) -> Path:
    return OPT_CONTRACT_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def contract_path(task: JsonDict, provider: str, model: str, source: str) -> Path:
    if require_source(source) == "raw":
        return raw_contract_path(task, provider, model)
    return optimized_contract_path(task, provider, model)


def generation_path(task: JsonDict, provider: str, model: str, contract_source: str = "raw") -> Path:
    if require_source(contract_source) == "raw":
        return output_path(RAW_OUT_DIR, task, provider, model, "contract_guided")
    return OPT_OUT_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: v for k, x in value.items() if (v := prune(x)) not in ("", None, [], {})}
    if isinstance(value, list):
        return [v for x in value if (v := prune(x)) not in ("", None, [], {})]
    return value


def take(items: Any, limit: int) -> list[Any]:
    return as_list(items)[:limit]


def source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def first_header_line(segment: str) -> str:
    return segment.strip().splitlines()[0].strip() if segment.strip() else ""


def function_header(segment: str) -> str:
    segment = segment.strip()
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
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return segment[:index].strip()

    return first_header_line(segment).rstrip(":")


def import_lines_from_task(task: JsonDict) -> list[str]:
    lines: list[str] = []
    raw = task.get("import_statement")

    if isinstance(raw, str):
        lines.extend(line.strip() for line in raw.splitlines() if line.strip())
    elif isinstance(raw, list):
        lines.extend(text(item) for item in raw if text(item))

    try:
        tree = ast.parse(skeleton(task))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                line = source_segment(skeleton(task), node).strip()
                if line:
                    lines.append(line)
    except SyntaxError:
        pass

    deduped: list[str] = []
    for line in lines:
        if line not in deduped:
            deduped.append(line)
    return deduped


def interface_from_skeleton(task: JsonDict) -> JsonDict:
    source = skeleton(task)
    interface: JsonDict = {
        "imports": import_lines_from_task(task),
        "class_name": class_name(task),
        "class_header": f"class {class_name(task)}:",
        "fields": as_list(task.get("fields")),
        "methods": [],
    }

    try:
        tree = ast.parse(source)
    except SyntaxError:
        interface["note"] = "Skeleton could not be parsed; use method metadata for required names."
        interface["methods"] = [
            {"method_name": text(method.get("method_name"))}
            for method in safe_methods_info(task)
            if text(method.get("method_name"))
        ]
        return prune(interface)

    cls = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name(task)),
        None,
    )
    if cls is None:
        return prune(interface)

    class_segment = source_segment(source, cls)
    interface["class_header"] = first_header_line(class_segment).rstrip(":") + ":"

    methods: list[JsonDict] = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        decorators = [
            source_segment(source, dec).strip()
            for dec in node.decorator_list
            if source_segment(source, dec).strip()
        ]
        header = function_header(source_segment(source, node))

        methods.append(
            prune(
                {
                    "method_name": node.name,
                    "decorators": decorators,
                    "signature": header,
                }
            )
        )

    interface["methods"] = methods
    return prune(interface)


def compact_callable(body: JsonDict) -> JsonDict:
    return prune(
        {
            "preconditions": take(body.get("preconditions"), 3),
            "postconditions": take(body.get("postconditions"), 5),
            "invariants": take(body.get("invariants"), 3),
            "edge_cases": take(body.get("edge_cases"), 5),
        }
    )


def compact_contract(contract: JsonDict) -> JsonDict:
    task = as_dict(contract.get("task"))
    constructor = as_dict(contract.get("constructor"))

    return prune(
        {
            "summary": text(task.get("summary"))[:1200],
            "constructor": {
                "signature": constructor.get("signature"),
                "initializes": constructor.get("initializes"),
                "contract": compact_callable(as_dict(constructor.get("contract"))),
            },
            "class_invariants": take(contract.get("class_invariants"), 6),
            "method_contracts": [
                prune(
                    {
                        "method_name": method.get("method_name"),
                        "signature": method.get("signature"),
                        "dependencies": method.get("dependencies"),
                        "contract": compact_callable(as_dict(method.get("contract"))),
                    }
                )
                for method in take(contract.get("method_contracts"), 30)
                if isinstance(method, dict)
            ],
            "interaction_contracts": take(contract.get("interaction_contracts"), 5),
        }
    )


def task_context(task: JsonDict) -> JsonDict:
    return prune(
        {
            "task_id": task_id(task),
            "class_name": class_name(task),
            "class_description": text(task.get("class_description"))[:3000],
            "class_constructor": text(task.get("class_constructor"))[:3000],
            "fields": as_list(task.get("fields")),
            "methods_info": safe_methods_info(task),
        }
    )


def contract_metadata(record: JsonDict, source: str, path: Path) -> JsonDict:
    meta: JsonDict = {
        "contract_path": str(path),
        "contract_status": record.get("status"),
        "contract_type": record.get("contract_type") or contract_type(source),
        "contract_version": record.get("contract_version") or contract_version(source),
    }

    if source == "optimized_rl":
        for key in (
            "optimization_status",
            "fallback_used",
            "issue",
            "reason",
            "contract_hash",
            "rl_policy",
        ):
            if key in record:
                meta[key] = record.get(key)

    return meta


def load_contract(task: JsonDict, provider: str, model: str, source: str) -> tuple[JsonDict, JsonDict]:
    path = contract_path(task, provider, model, source)

    if not path.exists():
        raise FileNotFoundError(f"Missing {source} contract file: {path}")

    record = load_json(path)

    if not isinstance(record, dict):
        raise ValueError(f"Contract record must be a JSON object: {path}")

    if record.get("status") != "success":
        raise RuntimeError(text(record.get("error")) or f"Contract generation failed: {path}")

    if not isinstance(record.get("contract"), dict):
        raise ValueError(f"Contract record has no contract object: {path}")

    return compact_contract(record["contract"]), contract_metadata(record, source, path)


def reusable_raw_code(task: JsonDict, provider: str, model: str, meta: JsonDict) -> tuple[str | None, str | None]:
    if meta.get("optimization_status") not in {
        "preserved_raw_passed",
        "no_optimized_candidate",
        "missing_feedback_raw_fallback",
    }:
        return None, None

    path = generation_path(task, provider, model, "raw")

    if not path.exists():
        return None, None

    record = load_json(path)
    code = record.get("generated_code") if isinstance(record, dict) else None

    if record.get("status") != "success" or not isinstance(code, str) or not code.strip():
        return None, None

    try:
        return extract_class_code(code, task, check_methods=True), str(path)
    except Exception:
        return None, None


def build_prompt(task: JsonDict, contract: JsonDict, meta: JsonDict, source: str) -> str:
    prompt = PROMPT.format(
        interface=pretty_json(interface_from_skeleton(task)),
        task_context=pretty_json(task_context(task)),
        contract_source=source,
        metadata=pretty_json(meta),
        contract=pretty_json(contract),
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
    reused_raw_generation: bool = False,
    raw_generation_path: str | None = None,
    **meta: Any,
) -> JsonDict:
    source = require_source(args.contract_source)
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model

    return {
        "task_id": task_id(task),
        "benchmark": BENCHMARK,
        "dataset": DATASET,
        "execution_model": EXECUTION_MODEL,
        "class_name": class_name(task),
        "entry_point": class_name(task),
        "stage": stage_for(source),
        "status": status,
        "provider": args.provider,
        "model_name": model,
        "contract_provider": contract_provider,
        "contract_model": contract_model,
        "contract_source": source,
        "contract_type": contract_type(source),
        "contract_version": contract_version(source),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "generation_prompt": prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "api_result": api_result,
        "api_latency_seconds": api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
        "reused_raw_generation": reused_raw_generation,
        "raw_generation_path": raw_generation_path,
        "error": error,
        "source": "ClassEval_data.json",
        **meta,
    }


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any) -> JsonDict:
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model

    contract: JsonDict | None = None
    meta: JsonDict = {}
    prompt = ""
    raw_response = ""
    api_result: JsonDict | None = None

    try:
        contract, meta = load_contract(task, contract_provider, contract_model, args.contract_source)

        if args.contract_source == "optimized_rl":
            code, path = reusable_raw_code(task, args.provider, model, meta)
            if code is not None:
                return make_record(
                    task,
                    args,
                    model,
                    status="success",
                    contract=contract,
                    prompt="Reused raw contract-guided generation because optimized contract preserved/fell back to raw.",
                    raw_response="",
                    generated_code=code,
                    api_result=None,
                    error=None,
                    reused_raw_generation=True,
                    raw_generation_path=path,
                    **meta,
                )

        prompt = build_prompt(task, contract, meta, args.contract_source)
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

        code = extract_class_code(raw_response, task, check_methods=True)

        return make_record(
            task,
            args,
            model,
            status="success",
            contract=contract,
            prompt=prompt,
            raw_response=raw_response,
            generated_code=code,
            api_result=api_result,
            error=None,
            **meta,
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
            **meta,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval code from raw or optimized contracts")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-source", choices=CONTRACT_SOURCES, default="raw")
    parser.add_argument("--contract-provider")
    parser.add_argument("--contract-model")
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
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = failed = skipped = reused = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = generation_path(task, args.provider, model, args.contract_source)

        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        record = generate_one(task, args, model, client)
        save_json(path, record)

        if record["status"] == "success":
            done += 1
            reused += int(record.get("reused_raw_generation") is True)
            suffix = " reused_raw" if record.get("reused_raw_generation") else ""
            print(f"[{index}] DONE {task_id(task)}{suffix}")
        else:
            failed += 1
            print(f"[{index}] FAILED {task_id(task)}: {record['error']}")

        if args.delay:
            time.sleep(args.delay)

    print(f"\nClassEval generation finished [{args.contract_source}].")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Reused raw generations: {reused}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


if __name__ == "__main__":
    main()