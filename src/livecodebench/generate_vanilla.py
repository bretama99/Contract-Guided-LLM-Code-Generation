from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_python_code
from src.common.task_utils import select_tasks


TASK_FILE = ROOT / "data" / "processed" / "livecodebench" / "livecodebench_tasks.json"
OUTPUT_DIR = OUTPUT_ROOT / "livecodebench" / "vanilla"
LOG_FILE = LOG_ROOT / "livecodebench_vanilla.log"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_DELAY = 1.0


def output_path(task_id: str, provider: str, model: str) -> Path:
    return (
        OUTPUT_DIR
        / safe_name(provider)
        / safe_name(model)
        / f"{safe_name(task_id)}_vanilla.json"
    )


def system_prompt() -> str:
    return (
        "You are a competitive programming Python code generator.\n"
        "Return ONLY complete valid Python source code.\n"
        "Do not use markdown fences.\n"
        "Do not explain.\n"
        "Do not include tests.\n"
        "Read input from stdin and write output to stdout.\n"
        "Prefer defining solve() and calling it under if __name__ == '__main__'.\n"
        "The output must parse with ast.parse."
    )


def user_prompt(task: dict[str, Any]) -> str:
    return str(task["prompt"]).strip()


def looks_truncated(text: str) -> bool:
    source = str(text or "").rstrip()

    if not source:
        return True

    bad_endings = (
        "+",
        "-",
        "*",
        "/",
        "%",
        "**",
        "//",
        "=",
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        ",",
        ".",
        ":",
        "(",
        "[",
        "{",
        "return",
        "if",
        "for",
        "while",
        "elif",
        "else",
    )

    last_line = source.splitlines()[-1].strip()

    if last_line in bad_endings:
        return True

    if re.search(r"(\bif\b|\bwhile\b|\bfor\b|\breturn\b|=|<|>|-|\+|\*)\s*$", last_line):
        return True

    if source.count("(") > source.count(")"):
        return True
    if source.count("[") > source.count("]"):
        return True
    if source.count("{") > source.count("}"):
        return True
    if source.count('"""') % 2 == 1:
        return True
    if source.count("'''") % 2 == 1:
        return True

    return False


def extract_or_raise(raw_response: str) -> str:
    code = extract_python_code(
        raw_response,
        entry_point=None,
        validate=True,
    )

    if looks_truncated(code):
        raise SyntaxError("model output appears truncated or incomplete")

    return code


def build_record(
    *,
    task: dict[str, Any],
    provider: str,
    model: str,
    prompt: str,
    raw_response: str,
    generated_code: str | None,
    status: str,
    temperature: float,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "benchmark": "livecodebench",
        "task_type": task.get("task_type", "stdin_stdout"),
        "entry_point": None,
        "stage": "vanilla",
        "status": status,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_used": prompt,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "error": error,
        "attempts": 1,
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
    prompt = user_prompt(task)
    raw_response = ""

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]

    try:
        raw_response, _usage = call_chat_model(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )

        generated_code = extract_or_raise(raw_response)

        return build_record(
            task=task,
            provider=provider,
            model=model,
            prompt=prompt,
            raw_response=raw_response,
            generated_code=generated_code,
            status="success",
            temperature=temperature,
        )

    except Exception as exc:
        return build_record(
            task=task,
            provider=provider,
            model=model,
            prompt=prompt,
            raw_response=raw_response,
            generated_code=None,
            status="failed",
            temperature=temperature,
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiveCodeBench Stage 1: vanilla generation"
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
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(
        load_json_list(TASK_FILE),
        start=args.start,
        count=args.count,
    )

    for offset, task in enumerate(tasks):
        index = args.start + offset
        task_id = str(task["task_id"])
        out_file = output_path(task_id, args.provider, model)
        if out_file.exists() and not args.overwrite:
            print(f"[{index}] SKIP {task_id}")
            continue
        record = generate_one(
            task=task,
            provider=args.provider,
            model=model,
            client=client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        save_json(out_file, record)
        if record["status"] == "success":
            print(f"[{index}] DONE {task_id} attempts=1")
            logging.info("Generated %s", task_id)
        else:
            print(f"[{index}] FAILED {task_id}: {record['error']}")
            logging.error("Failed %s: %s", task_id, record["error"])
        time.sleep(args.delay)
    print("LiveCodeBench vanilla generation finished.")

if __name__ == "__main__":
    main()