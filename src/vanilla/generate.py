import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import DATASETS, LOG_ROOT, OUTPUT_ROOT, ROOT
from src.common.io_utils import load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import (
    PROVIDERS,
    call_chat_model,
    default_model,
    get_client,
)
from src.common.parsing import extract_python_code
from src.common.task_utils import select_tasks


PROMPT_FILE = ROOT / "prompts" / "vanilla" / "base_prompt.txt"
OUTPUT_DIR = OUTPUT_ROOT / "vanilla"
LOG_FILE = LOG_ROOT / "vanilla.log"
DEFAULT_TEMPERATURE = 0.0


def load_prompt_template() -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")

    if "{prompt}" not in template:
        raise ValueError("Prompt template must contain {prompt}")

    return template


def vanilla_output_path(
    dataset: str,
    task_id: str,
    provider: str,
    model: str,
) -> Path:
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


def build_generation_record(
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
        "source": task.get("source"),
        "source_version": task.get("source_version"),
    }


def generate_one(
    *,
    task: dict[str, Any],
    provider: str,
    model: str,
    client,
    prompt_template: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = prompt_template.replace("{prompt}", task["prompt"])
    raw_response = ""

    try:
        messages = [
            {
                "role": "system",
                "content": "Generate only valid Python source code. Return code only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        raw_response, _speed = call_chat_model(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )

        generated_code = extract_python_code(
            raw_response,
            entry_point=task.get("entry_point"),
            validate=True,
        )

        return build_generation_record(
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
        return build_generation_record(
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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(LOG_FILE)

    dataset_path = DATASETS[args.dataset]["path"]
    model = args.model or default_model(args.provider)

    client = get_client(args.provider)
    prompt_template = load_prompt_template()

    all_tasks = load_json_list(dataset_path)
    tasks = select_tasks(all_tasks, start=args.start, count=args.count)

    for offset, task in enumerate(tasks):
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
            print(f"[{index}] DONE {task['task_id']}")
            logging.info("Generated %s", task["task_id"])
        else:
            print(f"[{index}] FAILED {task['task_id']}: {record['error']}")
            logging.error("Failed %s: %s", task["task_id"], record["error"])

        time.sleep(args.delay)

    print("Stage 1 vanilla generation finished.")

if __name__ == "__main__":
    main()