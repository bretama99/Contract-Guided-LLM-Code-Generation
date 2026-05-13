import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any
from datasets import load_dataset



ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "livecodebench" / "livecodebench_tasks.json"
SOURCE = "livecodebench/code_generation_lite"
VERSION = "release_v2"

def first(item: dict[str, Any], *keys: str) -> Any:
    return next((item[key] for key in keys if item.get(key)), None)

def task_id(item: dict[str, Any], index: int) -> str:
    value = str(first(item, "question_id", "task_id", "id", "problem_id") or index).strip()
    return value if value.startswith("LiveCodeBench/") else f"LiveCodeBench/{value}"

def build_prompt(item: dict[str, Any]) -> str:
    title = first(item, "question_title", "title", "name")
    content = first(item, "question_content", "prompt", "description", "problem_description")
    starter = first(item, "starter_code", "code", "function_signature")
    if not content:
        raise ValueError("No problem statement field found.")
    parts = [
        title,
        content,
        "Write a complete Python solution.",
        "Return only Python source code.",
        f"Starter code:\n{starter}" if starter else None,
    ]

    return "\n\n".join(str(part).strip() for part in parts if part)

def record(row: dict[str, Any], index: int) -> dict[str, Any]:
    item = dict(row)
    return {
        "task_id": task_id(item, index),
        "benchmark": "livecodebench",
        "entry_point": first(item, "entry_point", "function_name"),
        "prompt": build_prompt(item),
        "source": SOURCE,
        "source_version": f"{SOURCE}/{VERSION}", "raw": item,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench tasks")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    dataset = load_dataset(
        SOURCE,
        split="test",
        version_tag=VERSION,
        streaming=True,
        trust_remote_code=True,
    )

    records = [
        record(row, index)
        for index, row in enumerate(islice(dataset, args.limit))
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved {len(records)} LiveCodeBench tasks to {OUT}")

if __name__ == "__main__":
    main()