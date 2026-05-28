from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.contract_normalizer import normalize_contract_shape, stage2_shape_errors
from src.classeval.contract_schema import schema_for_prompt
from src.classeval.core import JsonDict, as_dict, as_list, task_id, text
from src.classeval.evaluate_contract_guided import evaluation_path as v1_evaluation_path
from src.classeval.generate_contracts import (
    DEFAULT_MAX_TOKENS,
    SYSTEM_PROMPT,
    TASK_FILE,
    contract_path as v1_contract_path,
)
from src.classeval.generate_from_contracts import generation_path as v1_generation_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.task_utils import select_tasks

STAGE = "2R_feedback_optimized_contract_generation"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contracts"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contracts.log"

logger = logging.getLogger(__name__)


def optimized_path(task: JsonDict, provider: str, model: str) -> Path:
    return OUT_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def optimized_generation_path(task: JsonDict, provider: str, model: str) -> Path:
    return (
        OUTPUT_ROOT
        / "classeval"
        / "rl_optimized_contract_guided"
        / safe_name(provider)
        / safe_name(model)
        / f"{safe_name(task_id(task))}.json"
    )


def optimized_evaluation_path(task: JsonDict, provider: str, model: str) -> Path:
    return (
        OUTPUT_ROOT
        / "classeval"
        / "evaluation"
        / "rl_optimized_contract_guided"
        / safe_name(provider)
        / safe_name(model)
        / f"{safe_name(task_id(task))}.json"
    )


def dump(value: Any, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def clip(value: Any, limit: int = 3000) -> str:
    value = value if isinstance(value, str) else dump(value)
    value = value.strip()
    return value[:limit] + ("...[truncated]" if len(value) > limit else "")


def safe_class_name(task: JsonDict) -> str:
    return text(task.get("class_name")) or text(task.get("entry_point")) or "UnknownClass"


def method_names(task: JsonDict) -> list[str]:
    names = [text(x) for x in as_list(task.get("method_names")) if text(x)]
    if names:
        return names

    return [
        text(x.get("method_name"))
        for x in as_list(task.get("methods_info"))
        if isinstance(x, dict) and text(x.get("method_name"))
    ]


def safe_methods_info(task: JsonDict) -> list[JsonDict]:
    blocked = ("test", "solution", "answer", "reference")
    out: list[JsonDict] = []

    for info in as_list(task.get("methods_info")):
        if not isinstance(info, dict):
            continue
        out.append({k: v for k, v in info.items() if not any(b in k.lower() for b in blocked)})

    return out


def task_view(task: JsonDict) -> JsonDict:
    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
        "entry_point": text(task.get("entry_point")) or safe_class_name(task),
        "class_description": text(task.get("class_description")),
        "class_constructor": text(task.get("class_constructor")),
        "fields": as_list(task.get("fields")),
        "imports": as_list(task.get("import_statement")),
        "method_names": method_names(task),
        "methods_info": safe_methods_info(task),
    }


def method_signature(info: JsonDict) -> str:
    for line in text(info.get("method_description")).splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line
    return ""


def method_summary(info: JsonDict) -> str:
    lines: list[str] = []

    for line in text(info.get("method_description")).replace('"""', "").replace("'''", "").splitlines():
        line = line.strip()
        if not line or line.startswith(("def ", ":param", ":return:", ">>>")):
            continue
        lines.append(line)

    return " ".join(lines).strip() or "Implements the behavior described by the task."


def dependencies(info: JsonDict) -> JsonDict:
    deps = as_dict(info.get("dependencies"))
    reads = as_list(deps.get("field_dependencies"))

    return {
        "reads": reads,
        "modifies": [],
        "preserves": reads,
        "calls": as_list(deps.get("method_dependencies")),
        "uses_libraries": as_list(deps.get("lib_dependencies")),
    }


def default_body(summary: str) -> JsonDict:
    return {
        "interface": {
            "inputs": [],
            "output": {
                "type": "",
                "description": summary,
            },
        },
        "preconditions": [],
        "postconditions": [summary],
        "invariants": [],
        "edge_cases": [],
        "invalid_input_behavior": {
            "specified": False,
            "expected_behavior": "",
            "exception_type": "",
            "description": "",
            "source": "",
        },
    }


def metadata_contract(task: JsonDict) -> JsonDict:
    methods: list[JsonDict] = []

    for info in as_list(task.get("methods_info")):
        if not isinstance(info, dict):
            continue

        name = text(info.get("method_name"))
        if not name:
            continue

        methods.append(
            {
                "method_name": name,
                "signature": method_signature(info),
                "dependencies": dependencies(info),
                "contract": default_body(method_summary(info)),
            }
        )

    return {
        "task": {
            "task_id": task_id(task),
            "benchmark": "ClassEval",
            "language": "python",
            "execution_model": "class_level",
            "class_name": safe_class_name(task),
            "entry_point": safe_class_name(task),
            "summary": text(task.get("class_description")),
        },
        "class_interface": {
            "fields": as_list(task.get("fields")),
            "methods": [m["method_name"] for m in methods],
        },
        "constructor": {
            "signature": "",
            "initializes": as_list(task.get("fields")),
            "contract": default_body("Initializes the class state required by the class."),
        },
        "class_invariants": [],
        "method_contracts": methods,
        "interaction_contracts": [],
    }


def repair_contract(contract: JsonDict, task: JsonDict) -> JsonDict:
    contract = normalize_contract_shape(contract)
    fallback = metadata_contract(task)

    task_block = as_dict(contract.get("task"))
    task_block.update(fallback["task"])
    contract["task"] = task_block

    contract["class_interface"] = as_dict(contract.get("class_interface")) or fallback["class_interface"]
    contract["constructor"] = as_dict(contract.get("constructor")) or fallback["constructor"]

    existing = {
        text(m.get("method_name")): m
        for m in as_list(contract.get("method_contracts"))
        if isinstance(m, dict) and text(m.get("method_name"))
    }

    repaired: list[JsonDict] = []

    for fallback_method in fallback["method_contracts"]:
        name = fallback_method["method_name"]
        method = as_dict(existing.get(name)).copy()
        body = as_dict(method.get("contract")).copy()
        fallback_body = fallback_method["contract"]

        method["method_name"] = name
        method["signature"] = text(method.get("signature")) or fallback_method["signature"]
        method["dependencies"] = as_dict(method.get("dependencies")) or fallback_method["dependencies"]

        interface = as_dict(body.get("interface")).copy()
        output = as_dict(interface.get("output")).copy()
        fallback_output = fallback_body["interface"]["output"]

        interface["inputs"] = as_list(interface.get("inputs"))
        output["type"] = text(output.get("type")) or fallback_output["type"]
        output["description"] = text(output.get("description")) or fallback_output["description"]
        interface["output"] = output

        body["interface"] = interface
        body["preconditions"] = as_list(body.get("preconditions"))
        body["postconditions"] = as_list(body.get("postconditions")) or fallback_body["postconditions"]
        body["invariants"] = as_list(body.get("invariants"))
        body["edge_cases"] = as_list(body.get("edge_cases"))
        body["invalid_input_behavior"] = (
            as_dict(body.get("invalid_input_behavior")) or fallback_body["invalid_input_behavior"]
        )

        method["contract"] = body
        repaired.append(method)

    contract["method_contracts"] = repaired
    contract["class_invariants"] = as_list(contract.get("class_invariants"))
    contract["interaction_contracts"] = as_list(contract.get("interaction_contracts"))

    return contract


def cleanup_json_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"```$", "", value).strip()
    value = value.replace("\ufeff", "")
    value = re.sub(r",\s*([}\]])", r"\1", value)
    return value


def json_candidates(value: str) -> list[str]:
    value = cleanup_json_text(value)
    candidates: list[str] = []

    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(value):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(value[start : i + 1])

    if value.startswith("{") and value not in candidates:
        candidates.insert(0, value)

    return candidates


def robust_json_loads(value: str) -> JsonDict:
    errors: list[str] = []

    for candidate in json_candidates(value):
        cleaned = cleanup_json_text(candidate)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("No valid JSON object found in model response. " + " | ".join(errors[:3]))


def parse_contract_response(response: str, task: JsonDict) -> tuple[JsonDict, list[str]]:
    raw = robust_json_loads(response)
    contract = repair_contract(raw, task)
    errors = stage2_shape_errors(contract, expected_methods=method_names(task))
    return contract, errors


def information_score(contract: JsonDict) -> int:
    score = len(as_list(contract.get("class_invariants"))) + len(as_list(contract.get("interaction_contracts")))

    for method in as_list(contract.get("method_contracts")):
        method = as_dict(method)
        body = as_dict(method.get("contract"))
        interface = as_dict(body.get("interface"))
        output = as_dict(interface.get("output"))
        deps = as_dict(method.get("dependencies"))

        score += int(bool(text(method.get("signature"))))
        score += int(bool(as_list(interface.get("inputs"))))
        score += int(bool(text(output.get("type"))))
        score += 2 * int(bool(text(output.get("description"))))
        score += 3 * int(bool(as_list(body.get("postconditions"))))
        score += 2 * int(bool(as_list(body.get("edge_cases"))))
        score += sum(
            int(bool(as_list(deps.get(k))))
            for k in ("reads", "modifies", "preserves", "calls", "uses_libraries")
        )

    return score


def load_previous_contract(
    task: JsonDict,
    provider: str,
    model: str,
    base_provider: str,
    base_model: str,
) -> tuple[JsonDict, str, str | None]:
    opt_path = optimized_path(task, provider, model)

    if opt_path.exists():
        record = load_json(opt_path)
        if record.get("status") == "success" and isinstance(record.get("contract"), dict):
            return repair_contract(record["contract"], task), "previous_optimized_contract", str(opt_path)

    v1_path = v1_contract_path(task, base_provider, base_model)

    if v1_path.exists():
        record = load_json(v1_path)
        if record.get("status") == "success" and isinstance(record.get("contract"), dict):
            return repair_contract(record["contract"], task), "original_stage2_contract", str(v1_path)

    return metadata_contract(task), "metadata_task_contract", None


def feedback_from_records(evaluation: JsonDict, generation: JsonDict | None = None) -> JsonDict:
    prepared = evaluation.get("feedback_for_next_contract")
    if isinstance(prepared, dict) and prepared:
        return prepared

    ev = as_dict(evaluation.get("evaluation"))
    metrics = as_dict(ev.get("metrics"))

    raw = "\n".join(
        text(x)
        for x in (
            evaluation.get("error"),
            evaluation.get("failure_type"),
            ev.get("stderr"),
            ev.get("error"),
            as_dict(generation or {}).get("error"),
        )
        if text(x)
    )

    markers = (
        "FAIL:",
        "ERROR:",
        "AssertionError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "ValueError",
        "Timeout",
        "expected",
        "got",
        "not found",
        "NoneType",
    )
    important = [line.strip() for line in raw.splitlines() if any(m in line for m in markers)]

    return {
        "passed": evaluation.get("passed") is True,
        "failure_type": text(evaluation.get("failure_type")),
        "tests_passed": metrics.get("passed"),
        "tests_total": metrics.get("total"),
        "failures": metrics.get("failures"),
        "errors": metrics.get("errors"),
        "failure_summary": clip("\n".join(important[:36]), 2600),
    }


def load_previous_feedback(
    task: JsonDict,
    provider: str,
    model: str,
    base_provider: str,
    base_model: str,
) -> tuple[JsonDict, str, str | None]:
    opt_eval = optimized_evaluation_path(task, provider, model)
    opt_gen = optimized_generation_path(task, provider, model)

    if opt_eval.exists():
        return (
            feedback_from_records(load_json(opt_eval), load_json(opt_gen) if opt_gen.exists() else None),
            "previous_optimized_evaluation",
            str(opt_eval),
        )

    v1_eval = v1_evaluation_path(task, base_provider, base_model)
    v1_gen = v1_generation_path(task, base_provider, base_model)

    if v1_eval.exists():
        return (
            feedback_from_records(load_json(v1_eval), load_json(v1_gen) if v1_gen.exists() else None),
            "original_contract_guided_evaluation",
            str(v1_eval),
        )

    return {}, "no_previous_feedback", None


def optimization_mode(contract_source: str, feedback: JsonDict) -> str:
    if contract_source == "metadata_task_contract":
        return "synthesis"
    if feedback.get("passed") is True:
        return "strengthening"
    return "correction"


def mode_instruction(mode: str) -> str:
    if mode == "synthesis":
        return (
            "No usable previous contract exists. Generate a complete optimized contract from the task. "
            "Focus on precise behavior, state changes, edge cases, and output guarantees."
        )

    if mode == "strengthening":
        return (
            "The previous generated code passed. Preserve passing behavior. Strengthen the contract by filling weak, "
            "empty, vague, or missing clauses. Add useful postconditions, invariants, edge cases, return-shape, "
            "ordering, formatting, and state guarantees. Do not add restrictive preconditions."
        )

    return (
        "The previous generated code failed. Correct the contract using feedback. Remove wrong assumptions, weaken "
        "preconditions that reject valid cases, and add missing boundary behavior, output guarantees, state mutation, "
        "formatting, ordering, and edge cases that prevent the observed failure."
    )


def prompt_contract_template(task: JsonDict) -> JsonDict:
    template = metadata_contract(task)
    template["method_contracts"] = [
        {
            "method_name": m["method_name"],
            "signature": m["signature"],
            "dependencies": m["dependencies"],
            "contract": {
                "interface": m["contract"]["interface"],
                "preconditions": [],
                "postconditions": ["..."],
                "invariants": [],
                "edge_cases": [],
                "invalid_input_behavior": m["contract"]["invalid_input_behavior"],
            },
        }
        for m in as_list(template.get("method_contracts"))
    ]
    return template


def make_prompt(
    task: JsonDict,
    previous: JsonDict,
    feedback: JsonDict,
    contract_source: str,
    feedback_source: str,
    mode: str,
) -> str:
    return "\n".join(
        [
            "Return exactly ONE valid minified JSON object. No markdown. No prose. No code.",
            "The JSON object is an optimized Stage-2 Design-by-Contract specification for one ClassEval Python class.",
            "",
            "OUTPUT_TEMPLATE_SHAPE:",
            dump(prompt_contract_template(task)),
            "",
            "OPTIMIZATION_MODE:",
            mode,
            "",
            "MODE_INSTRUCTION:",
            mode_instruction(mode),
            "",
            "TASK:",
            dump(task_view(task)),
            "",
            "PREVIOUS_CONTRACT_SOURCE:",
            contract_source,
            "",
            "PREVIOUS_CONTRACT:",
            dump(previous),
            "",
            "PREVIOUS_FEEDBACK_SOURCE:",
            feedback_source,
            "",
            "PREVIOUS_FEEDBACK:",
            dump(feedback),
            "",
            "RULES:",
            "- Output JSON only, with double quotes and no trailing commas.",
            "- Cover every visible method exactly once.",
            "- Preserve method names and signatures.",
            "- Improve passed contracts; correct failed contracts; synthesize missing contracts.",
            "- Do not skip revision because previous code passed.",
            "- Do not invent exceptions or strict invalid-input behavior unless supported by task or feedback.",
            "- Prefer concrete code-guiding clauses: return value, return type, state before/after, mutation, ordering, formatting, boundary cases.",
            "- If feedback shows a failure, encode the corrected expected behavior in the contract.",
        ]
    )


def repair_json_with_model(
    client: Any,
    model: str,
    bad_response: str,
    task: JsonDict,
    args: argparse.Namespace,
) -> tuple[JsonDict, str, JsonDict | None, list[str]]:
    prompt = "\n".join(
        [
            "The following model output was intended to be one JSON contract object, but it is invalid JSON.",
            "Repair only the JSON syntax. Preserve all contract content. Return JSON only.",
            "",
            "EXPECTED_METHODS:",
            dump(method_names(task)),
            "",
            "BAD_OUTPUT:",
            clip(bad_response, 12000),
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
        json_mode=True,
    )

    contract, errors = parse_contract_response(response, task)
    return contract, response, api_result, errors


def call_contract(
    client: Any,
    model: str,
    prompt: str,
    task: JsonDict,
    args: argparse.Namespace,
) -> tuple[JsonDict, str, JsonDict | None, list[str]]:
    last_error = ""
    last_response = ""
    last_api_result = None

    for attempt in range(max(1, args.format_retries + 1)):
        try:
            response, api_result = call_chat_model(
                client=client,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature if attempt == 0 else 0.0,
                max_tokens=args.max_tokens,
                json_mode=True,
            )
            last_response, last_api_result = response, api_result

            try:
                contract, errors = parse_contract_response(response, task)
                return contract, response, api_result, errors
            except Exception as parse_exc:
                last_error = str(parse_exc)
                contract, fixed_response, fixed_api_result, errors = repair_json_with_model(
                    client, model, response, task, args
                )
                return contract, fixed_response, fixed_api_result, errors

        except Exception as exc:
            last_error = str(exc)
            if attempt < args.format_retries:
                time.sleep(args.retry_delay)

    raise RuntimeError(last_error or "contract generation failed")


def make_record(task: JsonDict, args: argparse.Namespace, model: str, **extra: Any) -> JsonDict:
    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
        "stage": STAGE,
        "provider": args.provider,
        "model_name": model,
        "base_provider": args.base_provider or args.provider,
        "base_model": args.base_model or model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any) -> JsonDict:
    base_provider = args.base_provider or args.provider
    base_model = args.base_model or model

    previous, contract_source, contract_file = load_previous_contract(
        task,
        args.provider,
        model,
        base_provider,
        base_model,
    )
    feedback, feedback_source, feedback_file = load_previous_feedback(
        task,
        args.provider,
        model,
        base_provider,
        base_model,
    )

    mode = optimization_mode(contract_source, feedback)
    prompt = make_prompt(task, previous, feedback, contract_source, feedback_source, mode)

    try:
        contract, response, api_result, shape_errors = call_contract(client, model, prompt, task, args)
        status = "success"
        error = None
        fallback_to_previous = False
    except Exception as exc:
        logger.exception("Optimized contract model call failed; using previous contract: %s", task_id(task))
        contract = previous
        response = ""
        api_result = None
        shape_errors = [str(exc)]
        status = "success"
        error = str(exc)
        fallback_to_previous = True

    score = information_score(contract)

    return make_record(
        task,
        args,
        model,
        status=status,
        contract=contract,
        optimization_mode=mode,
        contract_source=contract_source,
        feedback_source=feedback_source,
        previous_contract_file=contract_file,
        previous_feedback_file=feedback_file,
        previous_information_score=information_score(previous),
        information_score=score,
        shape_errors=shape_errors,
        feedback=feedback,
        fallback_to_previous_contract=fallback_to_previous,
        error=error,
        **({"prompt": prompt, "raw_response": response, "api_result": api_result} if args.debug else {}),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one feedback-optimized ClassEval contract per task.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-provider")
    parser.add_argument("--base-model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--format-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = failed = skipped = fallback = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = optimized_path(task, args.provider, model)

        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        try:
            record = generate_one(task, args, model, client)
        except Exception as exc:
            logger.exception("Optimized contract failed: %s", task_id(task))
            record = make_record(task, args, model, status="failed", contract=None, error=str(exc))

        save_json(path, record)

        if record["status"] == "success":
            done += 1
            if record.get("fallback_to_previous_contract"):
                fallback += 1
            tag = " fallback=previous" if record.get("fallback_to_previous_contract") else ""
            print(
                f"[{index}] DONE {task_id(task)} "
                f"mode={record.get('optimization_mode')} "
                f"score={record.get('information_score')}{tag}"
            )
        else:
            failed += 1
            print(f"[{index}] FAILED {task_id(task)}: {record['error']}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval feedback-optimized contract generation finished.")
    print(f"Completed: {done}")
    print(f"Fallback-to-previous: {fallback}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()