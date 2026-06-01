from __future__ import annotations

from datasets import load_dataset

from src.common.config import DATASETS
from src.common.io_utils import save_json


SOURCE = "openai_humaneval"
SOURCE_VERSION = "openai_humaneval/test"


def to_record(item: dict) -> dict:
    return {
        "task_id": item["task_id"],
        "benchmark": "humaneval",
        "entry_point": item["entry_point"],
        "prompt": item["prompt"],
        "test": item["test"],
        "source": SOURCE,
        "source_version": SOURCE_VERSION,
    }


def main() -> None:
    dataset = load_dataset(SOURCE, split="test")
    records = [to_record(dict(item)) for item in dataset]

    output = DATASETS["humaneval"]["path"]
    save_json(output, records)

    print(f"Saved {len(records)} HumanEval tasks to {output}")


if __name__ == "__main__":
    main()