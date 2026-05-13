from datasets import load_dataset

from src.common.config import DATASETS
from src.common.io_utils import save_json


def main() -> None:
    dataset = load_dataset("openai_humaneval", split="test")
    records = []

    for item in dataset:
        records.append(
            {
                "task_id": item["task_id"],
                "benchmark": "humaneval",
                "entry_point": item["entry_point"],
                "prompt": item["prompt"],
                "test": item["test"],
                "canonical_solution": item["canonical_solution"],
                "source": "openai_humaneval",
                "source_version": "openai_humaneval/test",
            }
        )

    output_file = DATASETS["humaneval"]["path"]
    save_json(output_file, records)
    print(f"Saved {len(records)} HumanEval tasks to {output_file}")


if __name__ == "__main__":
    main()