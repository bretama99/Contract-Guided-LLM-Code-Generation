from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.classeval.core import (
    JsonDict,
    class_name,
    compact_json,
    ensure_no_reference_leak,
    extract_class_code,
    fill_template,
    method_profiles,
    output_path,
    pretty_json,
    read_template,
    safe_methods_info,
    skeleton,
    task_id,
)
from src.common.config import LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.task_utils import select_tasks

TASK_FILE = ROOT / "data" / "processed" / "classeval" / "ClassEval_data.json"
PLAN_PROMPT_FILE = ROOT / "prompts" / "classeval" / "agentic_plan_prompt.txt"
CODE_PROMPT_FILE = ROOT / "prompts" / "classeval" / "agentic_generate_prompt.txt"
OUT_DIR = OUTPUT_ROOT / "classeval" / "agentic_vanilla"
LOG_FILE = LOG_ROOT / "classeval_agentic_vanilla.log"

STAGE = "agentic_planned_vanilla"
PLAN_SYSTEM = "Return exactly one valid JSON planning object. Do not generate code."
CODE_SYSTEM = (
    "You are an expert Python code generation assistant. "
    "Return only complete, syntactically valid Python source code. "
    "Do not include markdown, explanations, or tests."
)

logger = logging.getLogger(__name__)


def task_structure(task: JsonDict) -> JsonDict:
    return {
        "task_id": task_id(task),
        "class_name": class_name(task),
        "class_description": task.get("class_description", ""),
        "constructor": task.get("class_constructor", ""),
        "fields": task.get("fields", []),
        "imports": task.get("import_statement", []),
        "method_names": list(method_profiles(skeleton(task), class_name(task))),
        "methods_info": safe_methods_info(task),
    }


def build_plan_prompt(task: JsonDict, template: str) -> str:
    prompt = fill_template(
        template,
        {
            "class_name": class_name(task),
            "structure": pretty_json(task_structure(task)),
            "skeleton": skeleton(task),
        },
    )
    ensure_no_reference_leak(task, prompt)
    return prompt


def build_code_prompt(task: JsonDict, plan: JsonDict, template: str) -> str:
    prompt = fill_template(
        template,
        {
            "class_name": class_name(task),
            "skeleton": skeleton(task),
            "plan": pretty_json(plan),
        },
    )
    ensure_no_reference_leak(task, prompt)
    return prompt


def parse_plan(raw: str) -> JsonDict:
    plan = extract_json_object(raw)

    if not isinstance(plan, dict):
        raise ValueError("Planning response is not a JSON object.")

    return {
        "class_summary": plan.get("class_summary", ""),
        "fields": plan.get("fields", []),
        "methods": plan.get("methods", []),
        "invariants": plan.get("invariants", []),
        "warnings": plan.get("warnings", []),
    }


def make_record(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    *,
    status: str,
    plan_prompt: str,
    plan_response: str,
    plan: JsonDict | None,
    code_prompt: str,
    raw_response: str,
    generated_code: str | None,
    plan_api_result: JsonDict | None,
    generation_api_result: JsonDict | None,
    error: str | None = None,
) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": "ClassEval",
        "dataset": "classeval",
        "execution_model": "class_level",
        "class_name": class_name(task),
        "entry_point": class_name(task),
        "stage": STAGE,
        "status": status,
        "provider": args.provider,
        "model_name": model,
        "temperature": args.temperature,
        "plan_tokens": args.plan_tokens,
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_prompt": plan_prompt,
        "plan_raw_response": plan_response,
        "plan": plan,
        "generation_prompt": code_prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "plan_api_result": plan_api_result,
        "generation_api_result": generation_api_result,
        "error": error,
        "source": "ClassEval_data.json",
    }


def generate_one(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    client: Any,
    plan_template: str,
    code_template: str,
) -> JsonDict:
    plan_prompt = plan_response = code_prompt = raw_response = ""
    plan: JsonDict | None = None
    plan_api: JsonDict | None = None
    code_api: JsonDict | None = None

    try:
        plan_prompt = build_plan_prompt(task, plan_template)
        plan_response, plan_api = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": plan_prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.plan_tokens,
            json_mode=True,
        )

        plan = parse_plan(plan_response)
        code_prompt = build_code_prompt(task, plan, code_template)

        raw_response, code_api = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": code_prompt},
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
            plan_prompt=plan_prompt,
            plan_response=plan_response,
            plan=plan,
            code_prompt=code_prompt,
            raw_response=raw_response,
            generated_code=extract_class_code(raw_response, task),
            plan_api_result=plan_api,
            generation_api_result=code_api,
        )

    except Exception as exc:
        logger.exception("Agentic vanilla generation failed for %s", task_id(task))
        return make_record(
            task,
            args,
            model,
            status="failed",
            plan_prompt=plan_prompt,
            plan_response=plan_response,
            plan=plan,
            code_prompt=code_prompt,
            raw_response=raw_response,
            generated_code=None,
            plan_api_result=plan_api,
            generation_api_result=code_api,
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval agentic vanilla code")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--plan-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8192)
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
    plan_template = read_template(PLAN_PROMPT_FILE, ("class_name", "structure", "skeleton"))
    code_template = read_template(CODE_PROMPT_FILE, ("class_name", "skeleton", "plan"))
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = failed = skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = output_path(OUT_DIR, task, args.provider, model, "agentic_vanilla")

        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        record = generate_one(task, args, model, client, plan_template, code_template)
        save_json(path, record)

        if record["status"] == "success":
            done += 1
            print(f"[{index}] DONE {task_id(task)}")
        else:
            failed += 1
            print(f"[{index}] FAILED {task_id(task)}: {record['error']}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval agentic vanilla generation finished.")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


if __name__ == "__main__":
    main()