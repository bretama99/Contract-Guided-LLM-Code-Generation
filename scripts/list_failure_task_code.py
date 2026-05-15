#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_task_id(task_cell: str) -> str:
    return str(task_cell).split(" ", 1)[0].strip()


def index_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["task_id"]): row
        for row in rows
        if row.get("task_id")
    }


def details_path(
    results_root: Path,
    method: str,
    dataset: str,
    provider: str,
    model: str,
) -> Path:
    return (
        results_root
        / "evalplus"
        / method
        / safe_name(provider)
        / safe_name(model)
        / dataset
        / "details.json"
    )


def is_failed(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("passed") is not True)


def get_code(row: dict[str, Any] | None) -> str:
    if not row:
        return ""

    generation = row.get("generation") or {}
    return str(
        generation.get("code")
        or generation.get("generated_code")
        or ""
    ).strip()


def get_entry_point(row: dict[str, Any] | None) -> str:
    if not row:
        return ""

    return str(
        row.get("entry_point")
        or row.get("task", {}).get("entry_point")
        or ""
    ).strip()


def get_evaluation(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}

    return row.get("evaluation") or {}


def make_failing_test(row: dict[str, Any] | None) -> str:
    if not row:
        return "# Missing result."

    generation = row.get("generation") or {}
    failure_type = row.get("failure_type")

    if generation.get("status") == "generation_failed" or failure_type == "generation_failed":
        return (
            "# Generation failed before evaluation\n"
            f"# reason: {generation.get('error') or failure_type}"
        )

    entry_point = get_entry_point(row)
    evaluation = get_evaluation(row)

    failed_case_index = evaluation.get("failed_case_index")
    failed_input = evaluation.get("input")
    expected = evaluation.get("expected")
    actual = evaluation.get("actual")
    actual_error = evaluation.get("actual_error")
    failure_type = evaluation.get("failure_type") or failure_type

    lines = []

    if failure_type is not None:
        lines.append(f"# failure_type: {failure_type}")

    if failed_case_index is not None:
        lines.append(f"# failed_case_index: {failed_case_index}")

    if failed_input is not None:
        lines.append(f"# input: {failed_input}")

    if expected is not None:
        lines.append(f"# expected: {expected}")

    if actual is not None:
        lines.append(f"# actual: {actual}")

    if actual_error:
        lines.append("# actual_error:")
        for line in str(actual_error).splitlines():
            lines.append(f"# {line}")

    lines.append("")

    if failed_input is not None and expected is not None:
        lines.append(f"assert {entry_point}(*{failed_input}) == {expected}")
    elif failed_input is not None:
        lines.append(f"{entry_point}(*{failed_input})")
    else:
        lines.append("# Failed test input was not available.")

    return "\n".join(lines)


def read_failure_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_markdown(args: argparse.Namespace) -> str:
    failure_rows = read_failure_csv(args.analysis_csv)

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}

    for row in failure_rows:
        dataset = row.get("Dataset", args.dataset or "evalplus")
        provider = row.get("Provider", args.provider or "")
        model = row.get("Model", args.model or "")

        if args.dataset and dataset != args.dataset:
            continue

        if args.provider and provider != args.provider:
            continue

        if args.model and model not in {args.model, safe_name(args.model)}:
            continue

        grouped.setdefault((dataset, provider, model), []).append(row)

    sections = []

    for (dataset, provider, model), rows in grouped.items():
        vanilla_file = details_path(
            args.results_root,
            "vanilla",
            dataset,
            provider,
            model,
        )
        contract_file = details_path(
            args.results_root,
            "raw_contracts",
            dataset,
            provider,
            model,
        )

        if not vanilla_file.exists():
            raise FileNotFoundError(f"Missing vanilla details: {vanilla_file}")

        if not contract_file.exists():
            raise FileNotFoundError(f"Missing contract-guided details: {contract_file}")

        vanilla_by_task = index_by_task(load_json(vanilla_file))
        contract_by_task = index_by_task(load_json(contract_file))

        for row in rows:
            task_id = parse_task_id(row["Task"])
            vanilla = vanilla_by_task.get(task_id)
            contract = contract_by_task.get(task_id)

            if not is_failed(vanilla) and not is_failed(contract):
                continue

            sections.append(f"# {task_id}")
            sections.append("")

            if is_failed(vanilla):
                sections.append("## Vanilla failing test")
                sections.append("```python")
                sections.append(make_failing_test(vanilla))
                sections.append("```")
                sections.append("")

                sections.append("## Vanilla failed code")
                sections.append("```python")
                sections.append(get_code(vanilla))
                sections.append("```")
                sections.append("")

            if is_failed(contract):
                sections.append("## Contract-guided failing test")
                sections.append("```python")
                sections.append(make_failing_test(contract))
                sections.append("```")
                sections.append("")

                sections.append("## Contract-guided failed code")
                sections.append("```python")
                sections.append(get_code(contract))
                sections.append("```")
                sections.append("")

            sections.append("---")
            sections.append("")

    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List only failed generated code and the failing test that caused the failure."
    )

    parser.add_argument(
        "--analysis-csv",
        required=True,
        type=Path,
        help="Failure-analysis CSV produced by build_failure_analysis_table.py.",
    )
    parser.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
    )
    parser.add_argument(
        "--out-md",
        default=Path("results/failure_analysis/failed_code_with_tests.md"),
        type=Path,
    )
    parser.add_argument("--dataset")
    parser.add_argument("--provider")
    parser.add_argument("--model")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text = build_markdown(args)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(text, encoding="utf-8")

    print(f"Wrote failed code and failing tests to: {args.out_md}")


if __name__ == "__main__":
    main()