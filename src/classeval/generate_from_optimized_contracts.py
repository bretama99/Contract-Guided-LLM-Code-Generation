from __future__ import annotations

import argparse
import ast
import json
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
    skeleton,
    task_id,
    text,
)
from src.classeval.generate_contracts import DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, TASK_FILE
from src.classeval.rl_generate_optimized_contracts import optimized_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_python_code
from src.common.task_utils import select_tasks

STAGE = "2R_optimized_contract_guided_generation"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contract_guided"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contract_guided.log"

logger = logging.getLogger(__name__)


def generation_path(task: JsonDict, provider: str, model: str) -> Path:
    return OUT_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def safe_class_name(task: JsonDict) -> str:
    return text(task.get("class_name")) or text(task.get("entry_point")) or class_name(task)


def keep(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value is True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def clean_list(value: Any) -> list[Any]:
    return [x for x in as_list(value) if keep(x)]


def add(dst: JsonDict, key: str, value: Any) -> None:
    if keep(value):
        dst[key] = value


def compact_input(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out: JsonDict = {}
    add(out, "name", text(value.get("name")))
    add(out, "type", text(value.get("type")))
    add(out, "description", text(value.get("description")))
    return out


def compact_output(value: Any) -> JsonDict:
    value = as_dict(value)
    out: JsonDict = {}
    add(out, "type", text(value.get("type")))
    add(out, "description", text(value.get("description")))
    return out


def compact_invalid_behavior(value: Any) -> JsonDict:
    value = as_dict(value)
    if value.get("specified") is not True:
        return {}

    out: JsonDict = {"specified": True}
    for key in ("expected_behavior", "exception_type", "description", "source"):
        add(out, key, text(value.get(key)))
    return out


def compact_method(method: JsonDict) -> JsonDict:
    body = as_dict(method.get("contract"))
    interface = as_dict(body.get("interface"))
    deps = as_dict(method.get("dependencies"))

    out: JsonDict = {
        "name": text(method.get("method_name")),
        "signature": text(method.get("signature")),
    }

    add(out, "inputs", [compact_input(x) for x in clean_list(interface.get("inputs"))])
    add(out, "returns", compact_output(interface.get("output")))

    for key in ("reads", "modifies", "preserves", "calls", "uses_libraries"):
        add(out, key, clean_list(deps.get(key)))

    add(out, "preconditions", clean_list(body.get("preconditions")))
    add(out, "postconditions", clean_list(body.get("postconditions")))
    add(out, "invariants", clean_list(body.get("invariants")))
    add(out, "edge_cases", clean_list(body.get("edge_cases")))
    add(out, "invalid_input_behavior", compact_invalid_behavior(body.get("invalid_input_behavior")))
    return out


def compact_contract(contract: JsonDict) -> JsonDict:
    task_block = as_dict(contract.get("task"))
    interface = as_dict(contract.get("class_interface"))
    constructor = as_dict(contract.get("constructor"))
    constructor_contract = as_dict(constructor.get("contract"))

    out: JsonDict = {
        "class_name": text(task_block.get("class_name")),
        "summary": text(task_block.get("summary")),
    }

    add(out, "fields", clean_list(interface.get("fields")))

    ctor: JsonDict = {}
    add(ctor, "signature", text(constructor.get("signature")))
    add(ctor, "initializes", clean_list(constructor.get("initializes")))
    add(ctor, "preconditions", clean_list(constructor_contract.get("preconditions")))
    add(ctor, "postconditions", clean_list(constructor_contract.get("postconditions")))
    add(ctor, "invariants", clean_list(constructor_contract.get("invariants")))
    add(ctor, "edge_cases", clean_list(constructor_contract.get("edge_cases")))
    add(out, "constructor", ctor)

    add(
        out,
        "methods",
        [compact_method(m) for m in clean_list(contract.get("method_contracts")) if isinstance(m, dict)],
    )
    add(out, "class_invariants", clean_list(contract.get("class_invariants")))
    add(out, "interaction_contracts", clean_list(contract.get("interaction_contracts")))
    return out


def method_descriptions(task: JsonDict) -> list[JsonDict]:
    out: list[JsonDict] = []
    blocked = ("test", "assert", "unittest", "expected", "candidate.py", "traceback")

    for info in as_list(task.get("methods_info")):
        if not isinstance(info, dict):
            continue

        name = text(info.get("method_name"))
        desc = text(info.get("method_description")).replace('"""', "").replace("'''", "")
        lines: list[str] = []

        for line in desc.splitlines():
            line = line.strip()
            low = line.lower()
            if not line or line.startswith(("def ", ">>>")):
                continue
            if any(b in low for b in blocked):
                continue
            lines.append(line)

        if name:
            out.append({"method": name, "description": " ".join(lines)[:1200]})
    return out


def strip_leaky_content(value: Any) -> Any:
    blocked_keys = (
        "test",
        "tests",
        "unit_test",
        "solution",
        "reference",
        "ground_truth",
        "oracle",
        "candidate",
        "stdout",
        "stderr",
        "traceback",
        "raw_response",
        "prompt",
        "api_result",
    )
    blocked_text = (
        "self.assert",
        "unittest",
        "def test_",
        "candidate.py",
        "traceback",
        "file \"/tmp",
        "__main__",
    )

    if isinstance(value, dict):
        out: JsonDict = {}
        for k, v in value.items():
            key = str(k)
            low = key.lower()
            if any(b in low for b in blocked_keys):
                continue
            out[key] = strip_leaky_content(v)
        return out

    if isinstance(value, list):
        cleaned = [strip_leaky_content(x) for x in value]
        return [x for x in cleaned if x not in ("", None, [], {})]

    if isinstance(value, str):
        low = value.lower()
        if any(b in low for b in blocked_text):
            return ""
        return value[:1800]

    return value


def contract_brief(contract: JsonDict) -> JsonDict:
    contract = strip_leaky_content(contract)
    out: JsonDict = {}

    add(out, "class_name", text(contract.get("class_name")))
    add(out, "summary", text(contract.get("summary")))
    add(out, "fields", clean_list(contract.get("fields")))
    add(out, "constructor", as_dict(contract.get("constructor")))
    add(out, "class_invariants", clean_list(contract.get("class_invariants")))
    add(out, "interaction_contracts", clean_list(contract.get("interaction_contracts")))

    methods: list[JsonDict] = []
    for method in clean_list(contract.get("methods")):
        if not isinstance(method, dict):
            continue
        m: JsonDict = {
            "name": text(method.get("name")),
            "signature": text(method.get("signature")),
        }
        add(m, "returns", as_dict(method.get("returns")))
        add(m, "reads", clean_list(method.get("reads")))
        add(m, "modifies", clean_list(method.get("modifies")))
        add(m, "postconditions", clean_list(method.get("postconditions")))
        add(m, "edge_cases", clean_list(method.get("edge_cases")))
        add(m, "invalid_input_behavior", as_dict(method.get("invalid_input_behavior")))
        methods.append(m)

    add(out, "methods", methods)
    return out


def load_contract(task: JsonDict, provider: str, model: str) -> JsonDict:
    path = optimized_path(task, provider, model)
    if not path.exists():
        raise FileNotFoundError(f"Missing optimized contract file: {path}")

    record = load_json(path)
    if record.get("status") != "success":
        raise RuntimeError(text(record.get("error")) or f"Optimized contract failed: {path}")

    contract = record.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Optimized contract record has no contract object: {path}")

    return compact_contract(contract)


def previous_feedback(task: JsonDict, provider: str, model: str) -> JsonDict:
    from src.classeval.evaluate_contract_guided_optimized import evaluation_path

    path = evaluation_path(task, provider, model)
    if not path.exists():
        return {}

    record = load_json(path)
    prepared = record.get("feedback_for_next_contract")

    if not isinstance(prepared, dict):
        prepared = {}

    # For code generation, keep only high-level feedback. Detailed failure lines may
    # include test names/assertions and trigger reference-leak protection.
    feedback: JsonDict = {}
    for key in ("passed", "failure_type", "tests_passed", "tests_total", "tests_failed", "failures", "errors", "skipped"):
        if prepared.get(key) not in (None, "", [], {}):
            feedback[key] = prepared.get(key)

    if feedback:
        return feedback

    evaluation = as_dict(record.get("evaluation"))
    metrics = as_dict(evaluation.get("metrics"))
    return {
        "passed": record.get("passed") is True,
        "failure_type": text(record.get("failure_type")),
        "tests_passed": metrics.get("passed"),
        "tests_total": metrics.get("total"),
        "failures": metrics.get("failures"),
        "errors": metrics.get("errors"),
    }


def build_prompt(payload: JsonDict) -> str:
    return "\n".join(
        [
            "Generate exactly ONE complete Python class for ClassEval.",
            "Return only Python code. No markdown. No explanation. No tests.",
            "",
            "INPUT:",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
            "RULES:",
            "- Preserve the class name and every method signature from the skeleton exactly.",
            "- Do not add @staticmethod unless it appears in the skeleton.",
            "- Do not rename, remove, or add parameters.",
            "- Every required method must contain executable code.",
            "- Use the optimized contract as the main semantic specification.",
            "- Use previous evaluation feedback only to avoid the same broad failure type.",
            "- If the task/skeleton and contract conflict, the task/skeleton are authoritative.",
            "- Do not copy previous V1 code.",
            "- Do not copy tests or prompt text.",
            "- Do not add artificial validation that rejects valid boundary cases unless explicitly required.",
            "- Do not raise exceptions unless the task or contract explicitly requires it.",
            "- Handle zero, empty strings, empty lists, empty dictionaries, missing keys, duplicates, boundary values, and None-like values when relevant.",
            "- Preserve required return type, return value, ordering, formatting, mutation, persistence, and side effects.",
            "- Do not use triple-quoted strings or docstrings.",
            "- Do not leave pass, ..., TODO, NotImplementedError, or empty method bodies.",
            "- Output a single Python source file.",
        ]
    )


def make_prompt(task: JsonDict, contract: JsonDict, feedback: JsonDict) -> str:
    payload: JsonDict = {
        "class_name": safe_class_name(task),
        "skeleton": skeleton(task),
        "imports": as_list(task.get("import_statement")),
        "class_description": text(task.get("class_description")).replace('"""', "").replace("'''", "").strip()[:1600],
        "method_descriptions": method_descriptions(task),
        "optimized_contract": strip_leaky_content(contract),
        "previous_optimized_evaluation_feedback": strip_leaky_content(feedback),
    }

    prompt = build_prompt(payload)
    try:
        ensure_no_reference_leak(task, prompt)
        return prompt
    except Exception:
        safe_payload: JsonDict = {
            "class_name": payload["class_name"],
            "skeleton": payload["skeleton"],
            "imports": payload["imports"],
            "class_description": "",
            "method_descriptions": [{"method": m["method"]} for m in payload["method_descriptions"]],
            "optimized_contract": contract_brief(contract),
            "previous_optimized_evaluation_feedback": {},
        }
        prompt = build_prompt(safe_payload)
        ensure_no_reference_leak(task, prompt)
        return prompt


def patch_empty_init(code: str, task: JsonDict) -> str:
    tree = ast.parse(code)
    expected = safe_class_name(task)

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != expected:
            continue

        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue

            docstrings = [
                stmt
                for stmt in item.body
                if isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ]
            executable = [stmt for stmt in item.body if stmt not in docstrings]

            if not executable or all(
                isinstance(stmt, ast.Pass)
                or (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                )
                for stmt in executable
            ):
                item.body = [ast.Return(value=ast.Constant(value=None))]
                ast.fix_missing_locations(tree)
                return ast.unparse(tree)

    return code


def extract_code(response: str, task: JsonDict) -> str:
    try:
        return extract_class_code(response, task, check_methods=True)
    except Exception:
        raw = extract_python_code(response, entry_point=None, validate=False).strip()
        patched = patch_empty_init(raw, task)
        return extract_class_code(patched, task, check_methods=True)


def repair_code_with_model(
    client: Any,
    model: str,
    bad_response: str,
    error: str,
    task: JsonDict,
    args: argparse.Namespace,
) -> tuple[str, str, JsonDict | None]:
    prompt = "\n".join(
        [
            "Repair the Python class below so it is valid and matches the required skeleton.",
            "Return only one complete Python source file. No markdown. No tests. No explanation.",
            "",
            "REQUIRED_SKELETON:",
            skeleton(task),
            "",
            "ERROR:",
            error[:2000],
            "",
            "BAD_OUTPUT:",
            bad_response[:12000],
            "",
            "RULES:",
            "- Preserve every method signature exactly.",
            "- Fix syntax, empty bodies, changed signatures, and missing methods.",
            "- Do not add tests.",
            "- Do not use triple-quoted strings or docstrings.",
        ]
    )

    response, api_result = call_chat_model(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=args.max_tokens,
        json_mode=False,
    )

    return extract_code(response, task), response, api_result


def call_model(client: Any, model: str, prompt: str, task: JsonDict, args: argparse.Namespace) -> tuple[str, str, JsonDict | None, bool]:
    response, api_result = call_chat_model(
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

    try:
        return extract_code(response, task), response, api_result, False
    except Exception as exc:
        if args.format_retries <= 0:
            raise
        code, repaired_response, repaired_api_result = repair_code_with_model(
            client=client,
            model=model,
            bad_response=response,
            error=str(exc),
            task=task,
            args=args,
        )
        return code, repaired_response, repaired_api_result, True


def make_record(task: JsonDict, args: argparse.Namespace, model: str, **extra: Any) -> JsonDict:
    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
        "stage": STAGE,
        "provider": args.provider,
        "model_name": model,
        "contract_provider": args.contract_provider or args.provider,
        "contract_model": args.contract_model or model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any) -> JsonDict:
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model

    contract = load_contract(task, contract_provider, contract_model)
    feedback = previous_feedback(task, args.provider, model) if args.use_previous_eval_feedback else {}
    prompt = make_prompt(task, contract, feedback)
    code, response, api_result, repaired = call_model(client, model, prompt, task, args)

    record = make_record(
        task,
        args,
        model,
        status="success",
        contract=contract,
        generated_code=code,
        copied_from_v1=False,
        format_repair_used=repaired,
        error=None,
    )

    if args.debug:
        record.update({"prompt": prompt, "raw_response": response, "api_result": api_result})

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one ClassEval solution from each optimized contract.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-provider")
    parser.add_argument("--contract-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--format-retries", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-previous-eval-feedback", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = failed = skipped = repaired = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = generation_path(task, args.provider, model)

        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        try:
            record = generate_one(task, args, model, client)
        except Exception as exc:
            logger.exception("Optimized-contract-guided generation failed: %s", task_id(task))
            record = make_record(
                task,
                args,
                model,
                status="failed",
                contract=None,
                generated_code=None,
                error=str(exc),
            )

        save_json(path, record)

        if record["status"] == "success":
            done += 1
            if record.get("format_repair_used"):
                repaired += 1
            suffix = " repaired" if record.get("format_repair_used") else ""
            print(f"[{index}] DONE {task_id(task)}{suffix}")
        else:
            failed += 1
            print(f"[{index}] FAILED {task_id(task)}: {record['error']}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval optimized-contract-guided generation finished.")
    print(f"Completed: {done}")
    print(f"Format-repaired: {repaired}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()