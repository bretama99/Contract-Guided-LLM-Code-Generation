import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "processed" / "bigcodebench" / "bigcodebench_tasks.json"

def choose_prompt(item: dict) -> tuple[str, str]:
    for key in ("instruct_prompt", "complete_prompt", "code_prompt"):
        prompt = item.get(key)
        if prompt:
            return prompt, key
    raise ValueError(f"No prompt found for task {item.get('task_id')}")

def main() -> None:
    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    records = []

    for item in dataset:
        prompt, prompt_type = choose_prompt(item)

        records.append(
            {
                "task_id": item["task_id"],
                "benchmark": "bigcodebench",
                "entry_point": item.get("entry_point"),
                "prompt": prompt,
                "prompt_type": prompt_type,
                "instruct_prompt": item.get("instruct_prompt"),
                "complete_prompt": item.get("complete_prompt"),
                "code_prompt": item.get("code_prompt"),
                "test": item.get("test"),
                "canonical_solution": item.get("canonical_solution"),
                "doc_struct": item.get("doc_struct"),
                "libs": item.get("libs"),
                "source": "bigcode/bigcodebench",
                "source_version": "v0.1.4",
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {len(records)} BigCodeBench tasks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()