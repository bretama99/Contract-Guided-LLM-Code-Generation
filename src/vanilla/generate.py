#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import DATASETS, LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_python_code
from src.common.task_utils import select_tasks, task_entry_point, task_identifier, task_prompt

STAGE = "vanilla"
PROMPT_FILE = ROOT / "prompts" / "vanilla" / "base_prompt.txt"
OUT_DIR = OUTPUT_ROOT / "vanilla"
LOG_FILE = LOG_ROOT / "vanilla.log"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_DELAY = 1.0

SYSTEM_PROMPT = (
    "You are an expert Python code generator. "
    "Return only complete, valid Python source code. "
    "Do not include markdown, explanations, or tests. "
    "Preserve the required function name and signature when visible."
)


def output_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return (
        OUT_DIR
        / safe_name(provider)
        / safe_name(model)
        / safe_name(dataset)
        / f"{safe_name(task_id)}_vanilla.json"
    )


def load_prompt_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Missing vanilla prompt template: {PROMPT_FILE}")

    template = PROMPT_FILE.read_text(encoding="utf-8")
    if "{prompt}" not in template:
        raise ValueError(f"Prompt template must contain {{prompt}}: {PROMPT_FILE}")

    return template


def looks_truncated(code: str) -> bool:
    code = (code or "").rstrip()
    if not code:
        return True

    last = code.splitlines()[-1].strip()
    bad_endings = {
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
    }

    return (
        last in bad_endings
        or re.search(r"(\bif\b|\bwhile\b|\bfor\b|\breturn\b|=|<|>|-|\+|\*)\s*$", last) is not None
        or code.count("(") > code.count(")")
        or code.count("[") > code.count("]")
        or code.count("{") > code.count("}")
        or code.count('"""') % 2 == 1
        or code.count("'''") % 2 == 1
    )


def extract_solution(raw_response: str, entry_point: str) -> str:
    code = extract_python_code(raw_response, entry_point=entry_point, validate=True)
    if looks_truncated(code):
        raise SyntaxError("Model output appears truncated or incomplete.")
    return code


def make_record(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    status: str,
    prompt_used: str,
    raw_response: str,
    generated_code: str | None,
    api_result: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "benchmark": benchmark,
        "dataset": dataset,
        "entry_point": task_entry_point(task),
        "stage": STAGE,
        "status": status,
        "provider": provider,
        "model_name": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_used": prompt_used,
        "raw_response": raw_response,
        "generated_code": generated_code,
        "api_result": api_result,
        "api_latency_seconds": api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
        "error": error,
        "attempts": 1,
        "source": task.get("source"),
        "source_version": task.get("source_version"),
    }


def generate_one(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    client: Any,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = prompt_template.replace("{prompt}", task_prompt(task))
    raw_response = ""
    api_result: dict[str, Any] | None = None
    generated_code: str | None = None

    try:
        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )

        generated_code = extract_solution(raw_response, task_entry_point(task))

        return make_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            status="success",
            prompt_used=prompt,
            raw_response=raw_response,
            generated_code=generated_code,
            api_result=api_result,
            error=None,
        )

    except Exception as exc:
        logging.exception("Vanilla generation failed for %s", task_identifier(task))
        return make_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            status="failed",
            prompt_used=prompt,
            raw_response=raw_response,
            generated_code=generated_code,
            api_result=api_result,
            error=str(exc),
        )


def run(args: argparse.Namespace) -> None:
    info = DATASETS[args.dataset]
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    template = load_prompt_template()

    tasks = select_tasks(load_json_list(info["path"]), start=args.start, count=args.count)

    counts = {"completed": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        tid = task_identifier(task)
        out = output_path(args.dataset, tid, args.provider, model)

        if out.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}] SKIP {tid}")
            continue

        record = generate_one(
            task=task,
            dataset=args.dataset,
            benchmark=info["label"],
            provider=args.provider,
            model=model,
            client=client,
            prompt_template=template,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        save_json(out, record)

        ok = record["status"] == "success"
        counts["completed" if ok else "failed"] += 1
        print(f"[{index}] {'DONE' if ok else 'FAILED'} {tid}" + ("" if ok else f": {record['error']}"))

        if args.delay > 0:
            time.sleep(args.delay)

    print("\nStage 1 vanilla generation finished")
    print(f"Dataset: {args.dataset} ({info['label']})")
    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Completed: {counts['completed']}")
    print(f"Failed: {counts['failed']}")
    print(f"Skipped: {counts['skipped']}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: vanilla code generation")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
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
    setup_logging(LOG_FILE)
    run(parse_args())


if __name__ == "__main__":
    main()