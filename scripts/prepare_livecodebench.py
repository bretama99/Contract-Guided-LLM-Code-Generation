from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import load_dataset

from src.common.config import DATASETS
from src.common.io_utils import save_json


SOURCE = "livecodebench/code_generation_lite"
VERSION = "release_v2"


def first(item: dict[str, Any], *keys: str) -> Any:
    return next((item[key] for key in keys if item.get(key)), None)


def make_task_id(item: dict[str, Any], index: int) -> str:
    value = str(first(item, "question_id", "task_id", "id", "problem_id") or index).strip()
    return value if value.startswith("LiveCodeBench/") else f"LiveCodeBench/{value}"


def parse_json_field(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def parse_public_tests(item: dict[str, Any]) -> dict[str, list[str]]:
    raw = first(item, "public_test_cases", "sample_test_cases", "examples", "test_cases")
    tests = parse_json_field(raw)

    if tests is None:
        return {"inputs": [], "outputs": []}

    if isinstance(tests, dict):
        tests = [tests]

    inputs: list[str] = []
    outputs: list[str] = []

    for case in tests if isinstance(tests, list) else []:
        if not isinstance(case, dict):
            continue

        kind = str(case.get("testtype") or case.get("type") or "stdin").lower()
        if kind not in {"stdin", "standard_input", "io"}:
            continue

        inp = case.get("input")
        out = case.get("output")
        if inp is None or out is None:
            continue

        inputs.append(str(inp))
        outputs.append(str(out))

    return {"inputs": inputs, "outputs": outputs}


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
        "Read from standard input and write to standard output.",
        "Do not define only a function unless the problem explicitly asks for one.",
        "Return only Python source code.",
        f"Starter code:\n{starter}" if starter else None,
    ]

    return "\n\n".join(str(part).strip() for part in parts if part)


def metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": item.get("platform"),
        "question_id": item.get("question_id"),
        "contest_id": item.get("contest_id"),
        "contest_date": item.get("contest_date"),
        "difficulty": item.get("difficulty"),
    }


def to_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    item = dict(row)

    return {
        "task_id": make_task_id(item, index),
        "benchmark": "livecodebench",
        "entry_point": first(item, "entry_point", "function_name") or "",
        "task_type": "stdin_stdout",
        "prompt": build_prompt(item),
        "input_output": parse_public_tests(item),
        "metadata": metadata(item),
        "source": SOURCE,
        "source_version": f"{SOURCE}/{VERSION}",
        "raw": item,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench tasks")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to save. If omitted, save all matching tasks.",
    )

    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Maximum number of raw rows to scan. If omitted, scan the full split.",
    )

    parser.add_argument(
        "--keep-missing-tests",
        action="store_true",
        help="Keep tasks even if public/sample tests are missing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        SOURCE,
        split="test",
        version_tag=VERSION,
        streaming=True,
        trust_remote_code=True,
    )

    records: list[dict[str, Any]] = []
    scanned = 0
    skipped_missing_tests = 0
    skipped_errors = 0

    for index, row in enumerate(dataset):
        if args.scan_limit is not None and scanned >= args.scan_limit:
            break

        scanned += 1

        try:
            item = to_record(row, index)
        except Exception as exc:
            skipped_errors += 1
            print(f"[skip:error] row={index} reason={exc}")
            continue

        tests = item.get("input_output", {})
        has_tests = bool(tests.get("inputs") and tests.get("outputs"))

        if not args.keep_missing_tests and not has_tests:
            skipped_missing_tests += 1
            continue

        records.append(item)

        if args.limit is not None and len(records) >= args.limit:
            break

    output = DATASETS["livecodebench"]["path"]
    save_json(output, records)

    print(f"Saved {len(records)} LiveCodeBench tasks to {output}")
    print(f"Scanned raw rows: {scanned}")
    print(f"Skipped missing tests: {skipped_missing_tests}")
    print(f"Skipped errors: {skipped_errors}")


if __name__ == "__main__":
    main()