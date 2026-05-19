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
    value = str(
        first(item, "question_id", "task_id", "id", "problem_id") or index
    ).strip()

    return value if value.startswith("LiveCodeBench/") else f"LiveCodeBench/{value}"


def build_prompt(item: dict[str, Any]) -> str:
    title = first(item, "question_title", "title", "name")
    content = first(
        item,
        "question_content",
        "prompt",
        "description",
        "problem_description",
    )
    starter = first(item, "starter_code", "code", "function_signature")

    if not content:
        raise ValueError("No problem statement field found.")

    parts = [
        title,
        content,
        "Write a complete Python solution.",
        "Read from standard input and write to standard output.",
        "Do not define only a function unless the problem explicitly asks for one.",
        "Return only Python source code.",
        f"Starter code:\n{starter}" if starter else None,
    ]

    return "\n\n".join(str(part).strip() for part in parts if part)

def parse_json_field(value: Any) -> Any:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_public_test_cases(item: dict[str, Any]) -> dict[str, list[str]]:
    raw_tests = first(
        item,
        "public_test_cases",
        "sample_test_cases",
        "examples",
        "test_cases",
    )

    tests = parse_json_field(raw_tests)

    if tests is None:
        return {
            "inputs": [],
            "outputs": [],
        }

    if isinstance(tests, dict):
        tests = [tests]

    inputs: list[str] = []
    outputs: list[str] = []

    for test in tests:
        if not isinstance(test, dict):
            continue

        test_type = str(test.get("testtype") or test.get("type") or "stdin").lower()

        if test_type not in {"stdin", "standard_input", "io"}:
            continue

        inp = test.get("input")
        out = test.get("output")

        if inp is None or out is None:
            continue

        inputs.append(str(inp))
        outputs.append(str(out))

    return {
        "inputs": inputs,
        "outputs": outputs,
    }


def metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": item.get("platform"),
        "question_id": item.get("question_id"),
        "contest_id": item.get("contest_id"),
        "contest_date": item.get("contest_date"),
        "difficulty": item.get("difficulty"),
    }


def record(row: dict[str, Any], index: int) -> dict[str, Any]:
    item = dict(row)

    return {
        "task_id": task_id(item, index),
        "benchmark": "livecodebench",
        "entry_point": first(item, "entry_point", "function_name"),
        "task_type": "stdin_stdout",
        "prompt": build_prompt(item),
        "input_output": parse_public_test_cases(item),
        "metadata": metadata(item),
        "source": SOURCE,
        "source_version": f"{SOURCE}/{VERSION}",
        "raw": item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench tasks")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of processed tasks to save after filtering.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=500,
        help="Maximum number of raw dataset rows to scan.",
    )
    parser.add_argument(
        "--keep-missing-tests",
        action="store_true",
        help="Keep tasks even when public stdin/stdout tests are missing.",
    )

    args = parser.parse_args()

    dataset = load_dataset(
        SOURCE,
        split="test",
        version_tag=VERSION,
        streaming=True,
        trust_remote_code=True,
    )

    records = []
    scanned = 0

    for index, row in enumerate(dataset):
        if scanned >= args.scan_limit:
            break

        scanned += 1
        item = record(row, index)

        has_tests = (
            item.get("input_output", {}).get("inputs")
            and item.get("input_output", {}).get("outputs")
        )

        if args.keep_missing_tests or has_tests:
            records.append(item)

        if len(records) >= args.limit:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(
        f"Saved {len(records)} LiveCodeBench tasks to {OUT} "
        f"(scanned {scanned} raw rows)"
    )
if __name__ == "__main__":
    main()