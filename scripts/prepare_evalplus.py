import argparse
import json
from pathlib import Path
from typing import Any

from src.common.config import DATASETS
from src.common.io_utils import save_json


LOADERS = {
    "humaneval": ("evalplus", "get_human_eval_plus"),
    "mbpp": ("evalplus_mbpp", "get_mbpp_plus"),
}


def load_evalplus(name: str) -> tuple[str, dict[str, dict[str, Any]]]:
    from evalplus import data

    dataset_key, loader_name = LOADERS[name]
    loader = getattr(data, loader_name)
    return dataset_key, loader()


def safe_json(value: Any) -> Any:
    """
    Converts EvalPlus objects into JSON-serializable values.
    Tuples become lists, and unusual objects become strings.
    """
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def to_record(
    *,
    task_id: str,
    problem: dict[str, Any],
    evalplus_dataset: str,
    include_tests: bool,
) -> dict[str, Any]:
    prompt = str(problem.get("prompt") or "").strip()

    if not prompt:
        raise ValueError(f"EvalPlus task has no prompt: {task_id}")

    base_input = safe_json(problem.get("base_input") or [])
    plus_input = safe_json(problem.get("plus_input") or [])

    record = {
        "task_id": task_id,
        "benchmark": "evalplus",
        "evalplus_dataset": evalplus_dataset,
        "entry_point": str(problem.get("entry_point") or "").strip(),
        "prompt": prompt,
        "source": "evalplus/evalplus",
        "source_version": f"evalplus/{evalplus_dataset}+",
        "evalplus_metadata": {
            "base_input_count": len(base_input),
            "plus_input_count": len(plus_input),
        },
    }

    if include_tests:
        record["base_input"] = base_input
        record["plus_input"] = plus_input

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EvalPlus tasks")

    parser.add_argument(
        "--evalplus-dataset",
        choices=sorted(LOADERS),
        default="humaneval",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")

    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include EvalPlus base_input and plus_input in the processed dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_key, problems = load_evalplus(args.evalplus_dataset)

    records = []

    for index, (task_id, problem) in enumerate(problems.items()):
        if args.limit is not None and index >= args.limit:
            break

        records.append(
            to_record(
                task_id=str(task_id),
                problem=safe_json(problem),
                evalplus_dataset=args.evalplus_dataset,
                include_tests=args.include_tests,
            )
        )

    output = Path(args.output) if args.output else DATASETS[project_key]["path"]
    save_json(output, records)

    print(f"Saved {len(records)} EvalPlus/{args.evalplus_dataset} tasks to {output}")
    print(f"Included tests: {args.include_tests}")


if __name__ == "__main__":
    main()