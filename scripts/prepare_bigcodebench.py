from __future__ import annotations

import argparse
from typing import Any

from datasets import load_dataset

from src.common.config import DATASETS
from src.common.io_utils import save_json


SOURCE = "bigcode/bigcodebench"
SOURCE_VERSION = "v0.1.4"


def choose_prompt(item: dict[str, Any]) -> tuple[str, str]:
    for key in ("instruct_prompt", "complete_prompt", "code_prompt"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    raise ValueError(f"No prompt found for task {item.get('task_id')}")


def to_record(item: dict[str, Any]) -> dict[str, Any]:
    prompt, prompt_type = choose_prompt(item)

    return {
        "task_id": item["task_id"],
        "benchmark": "bigcodebench",
        "entry_point": item.get("entry_point") or item.get("function_name") or "",
        "prompt": prompt,
        "prompt_type": prompt_type,
        "instruct_prompt": item.get("instruct_prompt"),
        "complete_prompt": item.get("complete_prompt"),
        "code_prompt": item.get("code_prompt"),
        "test": item.get("test"),
        "doc_struct": item.get("doc_struct"),
        "libs": item.get("libs"),
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BigCodeBench tasks")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(SOURCE, split=SOURCE_VERSION)

    records = []
    for index, item in enumerate(dataset):
        if args.limit is not None and index >= args.limit:
            break
        records.append(to_record(dict(item)))

    output = DATASETS["bigcodebench"]["path"]
    save_json(output, records)

    print(f"Saved {len(records)} BigCodeBench tasks to {output}")


if __name__ == "__main__":
    main()