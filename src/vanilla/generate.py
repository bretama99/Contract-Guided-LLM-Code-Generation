#!/usr/bin/env python3
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
from src.common.task_utils import select_tasks

PROMPT_FILE = ROOT / "prompts" / "vanilla" / "base_prompt.txt"
OUTPUT_DIR = OUTPUT_ROOT / "vanilla"
LOG_FILE = LOG_ROOT / "vanilla.log"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_DELAY = 1.0

def load_prompt_template() -> str:
    """Load the vanilla generation prompt template."""
    template = PROMPT_FILE.read_text(encoding="utf-8")
    if "{prompt}" not in template:
        raise ValueError(f"Prompt template must contain {{prompt}}: {PROMPT_FILE}")
    return template

def vanilla_output_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    """Return the output path for one vanilla generation record."""
    dataset_label = DATASETS[dataset]["label"]
    normalized_task_id = str(task_id).replace("\\", "/").strip()
    if not normalized_task_id.startswith(f"{dataset_label}/"):
        normalized_task_id = f"{dataset_label}/{normalized_task_id}"
    return (
        OUTPUT_DIR
        / safe_name(provider)
        / safe_name(model)
        / dataset
        / f"{safe_name(normalized_task_id)}_vanilla.json"
    )

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
    """Build a normalized generation record."""
    return {
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "entry_point": task.get("entry_point"),
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
    }

def system_prompt() -> str:
    """System instruction for vanilla code generation."""
    return (
        "You are a Python code generator.\n"
        "Return ONLY complete valid Python source code.\n"
        "Do not use markdown fences.\n"
        "Do not explain.\n"
        "Do not include tests.\n"
        "Do not stop early.\n"
        "Preserve the required function name and signature exactly.\n"
        "The output must parse with ast.parse."
    )

def looks_truncated(text: str) -> bool:
    """Heuristic check for obviously incomplete generated code."""
    source = (text or "").rstrip()
    if not source:
        return True
    bad_endings = ("+","-","*","/","%","**","//","=","==","!=","<","<=",">",">=",",",".",":","(","[",
                   "{","return","if","for","while","elif","else",
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

def extract_or_raise(raw_response: str, entry_point: str | None) -> str:
    """Extract Python code and reject incomplete outputs."""
    generated_code = extract_python_code(
        raw_response,
        entry_point=entry_point,
        validate=True,
    )
    if looks_truncated(generated_code):
        raise SyntaxError("model output appears truncated or incomplete")
    return generated_code

def generate_one(
    *,
    task: dict[str, Any],
    provider: str,
    model: str,
    client: Any,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Generate one vanilla solution with exactly one LLM call."""
    task_prompt = task["prompt"]
    prompt = prompt_template.replace("{prompt}", task_prompt)
    entry_point = task.get("entry_point")
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
        generated_code = extract_or_raise(raw_response, entry_point)
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
    args = parse_args()
    setup_logging(LOG_FILE)
    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    prompt_template = load_prompt_template()
    all_tasks = load_json_list(DATASETS[args.dataset]["path"])
    selected_tasks = select_tasks(all_tasks, start=args.start, count=args.count)
    for offset, task in enumerate(selected_tasks):
        index = args.start + offset
        output_file = vanilla_output_path(
            dataset=args.dataset,
            task_id=task["task_id"],
            provider=args.provider,
            model=model,
        )

        if output_file.exists() and not args.overwrite:
            print(f"[{index}] SKIP {task['task_id']}")
            continue

        record = generate_one(
            task=task,
            provider=args.provider,
            model=model,
            client=client,
            prompt_template=prompt_template,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        save_json(output_file, record)
        if record["status"] == "success":
            print(f"[{index}] DONE {task['task_id']} attempts=1")
            logging.info("Generated %s", task["task_id"])
        else:
            print(f"[{index}] FAILED {task['task_id']}: {record['error']}")
            logging.error("Failed %s: %s", task["task_id"], record["error"])
        time.sleep(args.delay)

    print("Stage 1 vanilla generation finished.")

if __name__ == "__main__":
    main()