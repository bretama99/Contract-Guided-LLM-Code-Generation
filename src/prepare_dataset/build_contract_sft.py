from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FINETUNING = HERE.parent
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, FINETUNING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.generation.generation_core import normalize_prompt_text
from src.prompts.generation_prompts import CONTRACT_SYSTEM_PROMPT
DIFFICULTIES = (
    "EASY",
    "MEDIUM",
    "MEDIUM_HARD",
    "HARD",
    "VERY_HARD",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )

            records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    temporary.replace(path)


def normalize_difficulty(value: Any) -> str:
    level = (
        str(value or "")
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )

    return {
        "MEDIUMHARD": "MEDIUM_HARD",
        "VERYHARD": "VERY_HARD",
    }.get(level, level)


def build_user_prompt(record: dict[str, Any]) -> str:
    contract = record["contract"]
    interface = contract.get("interface", {})
    mode = interface.get("mode")
    task_id = str(record["task_id"])
    question = str(record["original_prompt"]).strip()

    parts = [
        f"TASK_ID: {task_id}",
        "",
        "QUESTION:",
        question,
        "",
    ]

    if mode == "functional":
        parts.append("Use Call-Based format.")

        function_name = interface.get("function_name")
        if function_name:
            parts.append(f"FUNCTION NAME: {function_name}")
    else:
        parts.append("Use Standard Input format.")

    parts.extend(["", "CONTRACT:"])

    return normalize_prompt_text("\n".join(parts))


def build_example(record: dict[str, Any]) -> dict[str, Any]:
    contract = record["contract"]
    assistant = json.dumps(
        contract,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    json.loads(assistant)

    return {
        "task_id": str(record["task_id"]),
        "difficulty": normalize_difficulty(
            record.get("difficulty")
        ),
        "prompt_hash": str(record.get("prompt_hash") or ""),
        "number_of_test_cases": int(
            record.get("number_of_test_cases") or 0
        ),
        "test_cases_passed": int(
            record.get("test_cases_passed") or 0
        ),
        "messages": [
            {
                "role": "system",
                "content": normalize_prompt_text(
                    CONTRACT_SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": build_user_prompt(record),
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter cleaned TACO contracts and create "
            "stratified SFT train/validation files."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=51,
    )
    parser.add_argument(
        "--min-pass-ratio",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--all-cleaned",
        action="store_true",
        help="Use every cleaned contract and ignore execution thresholds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input_file.is_file():
        parser.error(
            f"Input file not found: {args.input_file}"
        )

    if args.min_tests < 1:
        parser.error("--min-tests must be positive.")

    if not 0 <= args.min_pass_ratio <= 1:
        parser.error(
            "--min-pass-ratio must be between 0 and 1."
        )

    if not 0 < args.val_ratio < 1:
        parser.error(
            "--val-ratio must be between 0 and 1."
        )

    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_file = args.output_dir / "selected_contracts.jsonl"
    train_file = args.output_dir / "train.jsonl"
    val_file = args.output_dir / "val.jsonl"
    summary_file = args.output_dir / "split_summary.json"

    outputs = (
        selected_file,
        train_file,
        val_file,
        summary_file,
    )

    if not args.overwrite:
        existing = [
            str(path)
            for path in outputs
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Outputs already exist:\n- "
                + "\n- ".join(existing)
            )

    source = load_jsonl(args.input_file)
    selected: list[dict[str, Any]] = []
    excluded = Counter()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for record in source:
        task_id = str(record.get("task_id") or "").strip()
        prompt_hash = str(record.get("prompt_hash") or "").strip()
        difficulty = normalize_difficulty(
            record.get("difficulty")
        )

        if not task_id:
            excluded["missing_task_id"] += 1
            continue

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID: {task_id}")

        if prompt_hash and prompt_hash in seen_hashes:
            raise ValueError(
                f"Duplicate prompt hash: {prompt_hash}"
            )

        if difficulty not in DIFFICULTIES:
            excluded["invalid_difficulty"] += 1
            continue

        total = int(record.get("number_of_test_cases") or 0)
        passed = int(record.get("test_cases_passed") or 0)
        ratio = passed / total if total else 0.0

        if not args.all_cleaned:
            if total < args.min_tests:
                excluded["insufficient_tests"] += 1
                continue

            if ratio < args.min_pass_ratio:
                excluded["low_pass_ratio"] += 1
                continue

        seen_ids.add(task_id)

        if prompt_hash:
            seen_hashes.add(prompt_hash)

        selected.append(record)

    if not selected:
        raise ValueError(
            "No contracts passed the selection criteria."
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in selected:
        groups[
            normalize_difficulty(record.get("difficulty"))
        ].append(record)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for offset, difficulty in enumerate(DIFFICULTIES):
        records = groups.get(difficulty, [])
        random.Random(
            args.seed + offset * 1_000_003
        ).shuffle(records)

        if len(records) <= 1:
            val_count = 0
        else:
            val_count = max(
                1,
                round(len(records) * args.val_ratio),
            )
            val_count = min(
                val_count,
                len(records) - 1,
            )

        val_part = records[:val_count]
        train_part = records[val_count:]

        train_records.extend(
            build_example(record)
            for record in train_part
        )
        val_records.extend(
            build_example(record)
            for record in val_part
        )

        split_counts[difficulty] = {
            "selected": len(records),
            "train": len(train_part),
            "validation": len(val_part),
        }

    random.Random(args.seed).shuffle(train_records)
    random.Random(args.seed + 1).shuffle(val_records)

    train_ids = {
        record["task_id"]
        for record in train_records
    }
    val_ids = {
        record["task_id"]
        for record in val_records
    }
    train_hashes = {
        record["prompt_hash"]
        for record in train_records
        if record["prompt_hash"]
    }
    val_hashes = {
        record["prompt_hash"]
        for record in val_records
        if record["prompt_hash"]
    }

    if train_ids & val_ids:
        raise ValueError(
            "Task-ID overlap between train and validation."
        )

    if train_hashes & val_hashes:
        raise ValueError(
            "Prompt overlap between train and validation."
        )

    write_jsonl(selected_file, selected)
    write_jsonl(train_file, train_records)
    write_jsonl(val_file, val_records)

    summary = {
        "input_file": str(args.input_file),
        "input_records": len(source),
        "selection": {
            "all_cleaned": args.all_cleaned,
            "minimum_tests": (
                None if args.all_cleaned else args.min_tests
            ),
            "minimum_pass_ratio": (
                None
                if args.all_cleaned
                else args.min_pass_ratio
            ),
            "selected": len(selected),
            "excluded": dict(sorted(excluded.items())),
        },
        "split": {
            "seed": args.seed,
            "validation_ratio": args.val_ratio,
            "train": len(train_records),
            "validation": len(val_records),
            "combined": (
                len(train_records) + len(val_records)
            ),
            "by_difficulty": split_counts,
        },
        "validation": {
            "task_id_overlap": len(train_ids & val_ids),
            "prompt_hash_overlap": len(
                train_hashes & val_hashes
            ),
            "accounting_valid": (
                len(selected)
                == len(train_records)
                + len(val_records)
            ),
        },
        "outputs": {
            "selected": str(selected_file),
            "train": str(train_file),
            "validation": str(val_file),
        },
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
