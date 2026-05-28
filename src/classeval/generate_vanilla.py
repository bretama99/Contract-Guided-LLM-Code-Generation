from __future__ import annotations
import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any
from src.classeval.core import (
    JsonDict,
    class_name,
    ensure_no_reference_leak,
    extract_class_code,
    fill_template,
    output_path,
    read_template,
    skeleton,
    task_id,
)
from src.common.config import LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json_list, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.task_utils import select_tasks

TASK_FILE = ROOT / "data" / "processed" / "classeval" / "ClassEval_data.json"
PROMPT_FILE = ROOT / "prompts" / "classeval" / "vanilla_prompt.txt"
OUT_DIR = OUTPUT_ROOT / "classeval" / "vanilla"
LOG_FILE = LOG_ROOT / "classeval_vanilla.log"
STAGE = "vanilla"
DEFAULT_MAX_TOKENS = 8192
SYSTEM_PROMPT = (
    "You are an expert Python class-level code generation assistant. "
    "Return only complete, syntactically valid Python source code. "
    "Do not include markdown, explanations, or tests."
)
logger = logging.getLogger(__name__)

def build_prompt(task: JsonDict, template: str) -> str:
    prompt = fill_template(template, {"skeleton": skeleton(task)})
    ensure_no_reference_leak(task, prompt)
    return prompt

def make_record(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    *,
    prompt: str,
    raw_response: str,
    generated_code: str | None,
    status: str,
    api_result: JsonDict | None,
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
        "max_tokens": args.max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_used": prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "api_result": api_result,
        "error": error,
        "source": "ClassEval_data.json",
    }

def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any, template: str) -> JsonDict:
    prompt = build_prompt(task, template)
    raw_response = ""
    api_result: JsonDict | None = None

    try:
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
            prompt=prompt,
            raw_response=raw_response,
            generated_code=extract_class_code(raw_response, task),
            status="success",
            api_result=api_result,
        )

    except Exception as exc:
        logger.exception("Vanilla generation failed for %s", task_id(task))
        return make_record(
            task,
            args,
            model,
            prompt=prompt,
            raw_response=raw_response,
            generated_code=None,
            status="failed",
            api_result=api_result,
            error=str(exc),
        )
         
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval vanilla code")
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
    template = read_template(PROMPT_FILE, ("skeleton",))
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)
    done = failed = skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = output_path(OUT_DIR, task, args.provider, model, "vanilla")
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

    print("\nClassEval vanilla generation finished.")
    print(f"Completed: {done}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")

if __name__ == "__main__":
    main()