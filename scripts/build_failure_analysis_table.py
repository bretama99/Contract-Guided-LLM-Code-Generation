#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


COLUMNS = [
    "Dataset",
    "Provider",
    "Model",
    "Task",
    "contract type",
    "reason why failed in vanilla",
    "reason why failed in contract guided",
    "error type",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> str:
    text = str(value or "").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", text)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")


def task_sort_key(task_id: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", task_id or "")
    return (
        re.sub(r"\d+$", "", task_id or ""),
        int(match.group(1)) if match else -1,
    )


def index_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in rows if row.get("task_id")}


def is_passed(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("passed") is True)


def failure_type(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing_result"

    return str(
        row.get("failure_type")
        or row.get("evaluation", {}).get("failure_type")
        or "unknown_failure"
    )


def failure_reason(row: dict[str, Any] | None) -> str:
    if not row:
        return "No result was found for this task."

    if row.get("passed") is True:
        return "Passed."

    if row.get("failure_explanation"):
        return clean(row["failure_explanation"])

    evaluation = row.get("evaluation") or {}
    generation = row.get("generation") or {}

    if generation.get("status") in {"generation_failed", "failed"} or generation.get("error"):
        return clean(f"Generation failed before evaluation: {generation.get('error')}.")

    failed_case_index = evaluation.get("failed_case_index")
    failed_input = evaluation.get("input")
    expected = evaluation.get("expected")
    actual = evaluation.get("actual")
    actual_error = evaluation.get("actual_error")

    if expected is not None and actual is not None:
        return clean(
            f"Failed test case #{failed_case_index}. "
            f"Input: {failed_input}. Expected: {expected}. Actual: {actual}."
        )

    if actual_error:
        return clean(
            f"Failed test case #{failed_case_index}. "
            f"Input: {failed_input}. Actual output could not be computed because: {actual_error}"
        )

    return f"Failed with error type: {failure_type(row)}."


def task_label(row: dict[str, Any] | None, task_id: str) -> str:
    entry_point = None

    if row:
        entry_point = row.get("entry_point") or row.get("task", {}).get("entry_point")

    return f"{task_id} ({entry_point})" if entry_point else task_id


def combined_error_type(
    vanilla: dict[str, Any] | None,
    contract_guided: dict[str, Any] | None,
) -> str:
    parts = []

    if not is_passed(vanilla):
        parts.append(f"vanilla={failure_type(vanilla)}")

    if not is_passed(contract_guided):
        parts.append(f"contract_guided={failure_type(contract_guided)}")

    return " | ".join(parts)


def non_empty_contract_field(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, dict):
        return len(value) > 0

    if isinstance(value, str):
        return bool(value.strip())

    return True


def contract_type_from_file(contract_path: Path | None) -> str:
    if not contract_path or not contract_path.exists():
        return "missing_contract"

    try:
        data = load_json(contract_path)
    except Exception:
        return "invalid_contract_file"

    if not isinstance(data, dict):
        return "invalid_contract_file"

    contract = data.get("contract", data)

    if not isinstance(contract, dict):
        return "invalid_contract_format"

    types = []

    if non_empty_contract_field(contract.get("preconditions")):
        types.append("precondition")

    if non_empty_contract_field(contract.get("postconditions")):
        types.append("postcondition")

    if non_empty_contract_field(contract.get("invariants")):
        types.append("invariant")

    return " + ".join(types) if types else "none"


def contract_task_id_from_file(path: Path) -> str | None:
    try:
        data = load_json(path)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    contract = data.get("contract", data)

    if isinstance(contract, dict):
        task = contract.get("task")
        if isinstance(task, dict) and task.get("task_id"):
            return str(task["task_id"])

        if contract.get("task_id"):
            return str(contract["task_id"])

    if data.get("task_id"):
        return str(data["task_id"])

    return None


def build_contract_index(contract_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}

    if not contract_dir.exists():
        return index

    for path in sorted(contract_dir.glob("*_contract.json")):
        task_id = contract_task_id_from_file(path)

        if task_id:
            index[task_id] = path
            index[safe_name(task_id)] = path

        stem = path.name.removesuffix("_contract.json")
        index[stem] = path
        index[safe_name(stem)] = path

    return index


def find_contract_file(
    *,
    task_id: str,
    contract_dir: Path,
    contract_index: dict[str, Path],
) -> Path | None:
    keys = [
        task_id,
        safe_name(task_id),
        task_id.replace("/", "_"),
    ]

    for key in keys:
        if key in contract_index:
            return contract_index[key]

    direct = contract_dir / f"{safe_name(task_id)}_contract.json"
    if direct.exists():
        return direct

    return None


def load_metadata(details_path: Path, fallback: dict[str, str]) -> dict[str, str]:
    metadata = dict(fallback)
    summary_path = details_path.parent / "summary.json"

    if not summary_path.exists():
        return metadata

    try:
        summary = load_json(summary_path)
    except Exception:
        return metadata

    if not isinstance(summary, dict):
        return metadata

    for key in ("dataset", "provider", "model"):
        if summary.get(key):
            metadata[key] = str(summary[key])

    return metadata


def discover_evalplus_pairs(results_root: Path) -> list[dict[str, Any]]:
    pairs = []

    vanilla_root = results_root / "evalplus" / "vanilla"
    raw_root = results_root / "evalplus" / "raw_contracts"

    if not vanilla_root.exists():
        return pairs

    for vanilla_details in sorted(vanilla_root.glob("*/*/*/details.json")):
        provider_folder = vanilla_details.parts[-4]
        model_folder = vanilla_details.parts[-3]
        dataset_folder = vanilla_details.parts[-2]

        contract_details = raw_root / provider_folder / model_folder / dataset_folder / "details.json"

        if not contract_details.exists():
            continue

        metadata = load_metadata(
            vanilla_details,
            {
                "dataset": dataset_folder,
                "provider": provider_folder,
                "model": model_folder,
            },
        )
        metadata = load_metadata(contract_details, metadata)

        pairs.append(
            {
                "dataset": metadata["dataset"],
                "provider": metadata["provider"],
                "model": metadata["model"],
                "safe_dataset": dataset_folder,
                "safe_provider": provider_folder,
                "safe_model": model_folder,
                "vanilla_details": vanilla_details,
                "contract_details": contract_details,
            }
        )

    return pairs


def build_rows_for_pair(
    *,
    pair: dict[str, Any],
    outputs_root: Path,
) -> list[dict[str, str]]:
    vanilla_rows = load_json(pair["vanilla_details"])
    contract_rows = load_json(pair["contract_details"])

    if not isinstance(vanilla_rows, list):
        raise TypeError(f"Vanilla details must be a JSON list: {pair['vanilla_details']}")

    if not isinstance(contract_rows, list):
        raise TypeError(f"Contract-guided details must be a JSON list: {pair['contract_details']}")

    vanilla_by_task = index_by_task(vanilla_rows)
    contract_by_task = index_by_task(contract_rows)

    contract_dir = (
        outputs_root
        / "contracts"
        / "raw"
        / pair["safe_provider"]
        / pair["safe_model"]
        / pair["safe_dataset"]
    )
    contract_index = build_contract_index(contract_dir)

    rows = []

    task_ids = sorted(
        set(vanilla_by_task) | set(contract_by_task),
        key=task_sort_key,
    )

    for task_id in task_ids:
        vanilla = vanilla_by_task.get(task_id)
        contract_guided = contract_by_task.get(task_id)

        if is_passed(vanilla) and is_passed(contract_guided):
            continue

        contract_file = find_contract_file(
            task_id=task_id,
            contract_dir=contract_dir,
            contract_index=contract_index,
        )

        rows.append(
            {
                "Dataset": pair["dataset"],
                "Provider": pair["provider"],
                "Model": pair["model"],
                "Task": task_label(contract_guided or vanilla, task_id),
                "contract type": contract_type_from_file(contract_file),
                "reason why failed in vanilla": failure_reason(vanilla),
                "reason why failed in contract guided": failure_reason(contract_guided),
                "error type": combined_error_type(vanilla, contract_guided),
            }
        )

    return rows


def auto_output_path(args: argparse.Namespace) -> Path:
    parts = []

    if args.dataset:
        parts.append(safe_name(args.dataset))

    if args.provider:
        parts.append(safe_name(args.provider))

    if args.model:
        parts.append(safe_name(args.model))

    filename = "_".join(parts) + "_failure_analysis.csv" if parts else "all_models_failure_analysis.csv"

    return args.results_root / "failure_analysis" / filename


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    return Path("/tmp") / path.name


def write_csv(rows: list[dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    target = next_available_path(path)

    try:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return target

    except PermissionError:
        fallback = next_available_path(Path("/tmp") / path.name)

        with fallback.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        return fallback


def write_markdown(rows: list[dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(value: str) -> str:
        return clean(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(cell(row[column]) for column in COLUMNS) + " |")

    target = next_available_path(path)

    try:
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    except PermissionError:
        fallback = next_available_path(Path("/tmp") / path.name)
        fallback.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return fallback


def keep_pair(pair: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.dataset and args.dataset not in {pair["dataset"], pair["safe_dataset"]}:
        return False

    if args.provider and args.provider not in {pair["provider"], pair["safe_provider"]}:
        return False

    if args.model and args.model not in {pair["model"], pair["safe_model"]}:
        return False

    return True


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_contract_type: dict[str, int] = {}
    by_provider_model: dict[str, int] = {}

    for row in rows:
        contract_type = row["contract type"]
        provider_model = f"{row['Provider']} | {row['Model']} | {row['Dataset']}"

        by_contract_type[contract_type] = by_contract_type.get(contract_type, 0) + 1
        by_provider_model[provider_model] = by_provider_model.get(provider_model, 0) + 1

    return {
        "total_rows": len(rows),
        "by_contract_type": by_contract_type,
        "by_provider_model": by_provider_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build failure-analysis table for vanilla vs raw-contract-guided results."
    )

    parser.add_argument("--results-root", default=Path("results"), type=Path)
    parser.add_argument("--outputs-root", default=Path("outputs"), type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--out-csv", default=None, type=Path)
    parser.add_argument("--out-md", default=None, type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = args.out_csv or auto_output_path(args)

    pairs = [
        pair
        for pair in discover_evalplus_pairs(args.results_root)
        if keep_pair(pair, args)
    ]

    if not pairs:
        raise SystemExit(
            f"No matching vanilla/raw_contracts pairs found under {args.results_root}."
        )

    rows = []

    for pair in pairs:
        rows.extend(
            build_rows_for_pair(
                pair=pair,
                outputs_root=args.outputs_root,
            )
        )

    csv_path = write_csv(rows, out_csv)

    print(f"Compared {len(pairs)} provider/model/dataset pairs.")
    print(f"Wrote {len(rows)} failure rows to {csv_path}")

    if args.out_md:
        md_path = write_markdown(rows, args.out_md)
        print(f"Wrote markdown table to {md_path}")

    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()