import argparse, re, json
from functools import lru_cache
from pathlib import Path
from typing import Any
import pandas as pd
from src.common.parsing import extract_json_object
from src.common.llm_clients import call_chat_model, get_client
from src.common.config import DATASETS
from analysis_tools.analyze_results import HELPED, REGRESSED, FAILED
from analysis_tools.contract_analysis_scripts.specs import AnalysisSpec

DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "google/gemini-3.5-flash"
MAX_LLM_ATTEMPTS = 3

SUMMARY_SHEET = "Summary"
INPUT_ERRORS_SHEET = "Input-Related Errors"
OUTPUT_ERRORS_SHEET = "Output-Related Errors"
OTHER_ERRORS_SHEET = "Other Errors"

BASE_DETAIL_COLUMNS = [
    "Task ID",
    "Prompt",
    "Vanilla Code",
    "Contract-Guided Code",
    "Contract",
]

COMMON_PROMPT_PARTS = {
    "failure_reasons": "common/failure_reasons.txt",
    "contract_quality_definitions": "common/contract_quality.txt",
    "yes_no_rule": "common/yes_no_rule.txt",
}

DEFAULT_FAILURE_REASON_BUCKETS = {
    "input interface/parsing mismatch": "input",
    "invalid input assumption": "input",
    "missing input case handling": "input",

    "wrong output value": "output",
    "incomplete or extra output": "output",
    "wrong output format/type": "output",
}

YES_NO_UNCLEAR = {"yes", "no", "unclear"}

VALID_FAILURE_REASONS = {
    "input interface/parsing mismatch",
    "invalid input assumption",
    "missing input case handling",
    "wrong output value",
    "incomplete or extra output",
    "wrong output format/type",
    "algorithmic logic error",
    "inefficient algorithm",
    "non-executable or missing implementation",
    "unclear",
}


def parse_args() -> tuple[Path, str, str, bool, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--contract-version",
        choices=["raw_contracts", "optimized_rl"],
        default="raw_contracts",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--helped", action="store_true")
    group.add_argument("--regressed", action="store_true")
    group.add_argument("--failed", action="store_true")

    args = parser.parse_args()

    if args.helped:
        analysis_type = "helped"
    elif args.regressed:
        analysis_type = "regressed"
    elif args.failed:
        analysis_type = "failed"
    else:
        raise ValueError("Expected one analysis type.")

    return (
        Path(args.root).resolve(),
        args.dataset,
        args.model,
        args.overwrite,
        analysis_type,
        args.contract_version,
    )


def read_input_sheet(
    root: Path,
    dataset: str,
    model: str,
    analysis_type: str,
    contract_version: str,
) -> pd.DataFrame:
    analysis_type = analysis_type.strip().lower()

    path = (
        root
        / "analysis"
        / safe(model)
        / safe(dataset)
        / "basic"
        / f"{safe(contract_version)}_analysis.xlsx"
    )

    if not path.exists():
        raise FileNotFoundError(f"{dataset}: missing analysis file: {path}")

    if analysis_type == "helped":
        sheet_name = HELPED
    elif analysis_type == "regressed":
        sheet_name = REGRESSED
    elif analysis_type == "failed":
        sheet_name = FAILED
    else:
        raise ValueError("Invalid analysis type in script. Cannot read input sheet.")

    return pd.read_excel(path, sheet_name=sheet_name)


def get_output_path(
    root: Path,
    dataset: str,
    model: str,
    analysis_type: str,
    contract_version: str,
) -> Path:
    analysis_type = analysis_type.strip().lower()

    if analysis_type not in ("helped", "regressed", "failed"):
        raise ValueError("Invalid analysis type in script. Cannot write output.")

    out = (
        root
        / "analysis"
        / safe(model)
        / safe(dataset)
        / "contracts"
        / safe(contract_version)
        / f"{analysis_type}_analysis.xlsx"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def get_stored_analysis_path(
    root: Path,
    dataset: str,
    model: str,
    task_id: Any,
    analysis_type: str,
    contract_version: str,
) -> Path:
    task_id = safe(str(task_id))
    analysis_type = analysis_type.strip().lower()

    if analysis_type not in ("helped", "regressed", "failed"):
        raise ValueError("Invalid analysis type in script. Cannot get stored analysis path.")

    path = (
        root
        / "analysis"
        / safe(model)
        / safe(dataset)
        / "contracts"
        / safe(contract_version)
        / "log"
        / analysis_type
        / f"{task_id}_{analysis_type}_analysis.json"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_stored_analysis_json(log_path: Path) -> dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Stored analysis is not a JSON object: {log_path}")

    return data


def write_stored_analysis_json(log_path: Path, analysis: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, sort_keys=True)
        f.write("\n")


@lru_cache(maxsize=None)
def read_prompt_template(filename: str) -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / filename
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_common_prompt_parts() -> dict[str, str]:
    return {
        name: read_prompt_template(filename).strip()
        for name, filename in COMMON_PROMPT_PARTS.items()
    }


@lru_cache(maxsize=None)
def get_cached_client(provider: str):
    return get_client(provider)


def get_llm_response(prompt: str) -> dict[str, Any]:
    raw_response, _ = call_chat_model(
        client=get_cached_client(DEFAULT_PROVIDER),
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=3000,
        json_mode=True,
    )

    data = extract_json_object(raw_response)
    if not isinstance(data, dict):
        raise ValueError("Parsed response must be a JSON object")
    return data


def get_task_details(task_row: pd.Series) -> dict[str, Any]:
    return {
        "Task ID": task_row["Task ID"],
        "Prompt": task_row["Prompt"],
        "Vanilla Code": task_row["Vanilla Code"],
        "Vanilla Failing Test": task_row.get("Vanilla Failing Test", None),
        "Contract-Guided Code": task_row["Contract-Guided Code"],
        "Contract-Guided Failing Test": task_row.get("Contract-Guided Failing Test", None),
        "Contract": get_contract_string(task_row),
        "Has Precondition": not is_missing(task_row["Preconditions"]),
        "Has Postcondition": not is_missing(task_row["Postconditions"]),
        "Has Invariant": not is_missing(task_row["Invariants"]),
    }


def get_contract_string(row: pd.Series) -> str:
    preconditions = clean_cell(row["Preconditions"])
    postconditions = clean_cell(row["Postconditions"])
    invariants = clean_cell(row["Invariants"])

    return "\n\n".join(
        [
            f"Preconditions: {preconditions}",
            f"Postconditions: {postconditions}",
            f"Invariants: {invariants}",
        ]
    )


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def clean_yes_no_unclear(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Field {field_name!r} must be a string.")

    value = value.strip().lower()
    if value not in YES_NO_UNCLEAR:
        raise ValueError(
            f"Field {field_name!r} must be one of {sorted(YES_NO_UNCLEAR)}, got {value!r}."
        )

    return value


def clean_first_failure_reason(value: Any) -> str:
    if value is None:
        return "unclear"

    if isinstance(value, list):
        if not value:
            return "unclear"
        value = value[0]

    if not isinstance(value, str):
        value = str(value)

    value = value.strip().lower()
    return value or "unclear"


def normalize_label(value: Any) -> str:
    if value is None:
        return "unclear"
    return str(value).strip().lower() or "unclear"


def get_causal_status(entry: dict[str, Any], spec: AnalysisSpec) -> str:
    if spec.effect_output_column:
        effect_value = normalize_label(entry.get(spec.effect_output_column))

        if effect_value in spec.causal_positive_effects:
            return "yes"

        if effect_value in spec.causal_negative_effects:
            return "no"

        return "unclear"

    if spec.causal_output_column:
        return normalize_label(entry.get(spec.causal_output_column))

    return "unclear"


def get_bucket_failure_reason(entry: dict[str, Any], spec: AnalysisSpec) -> str:
    if spec.failure_reason_output_columns:
        return normalize_label(entry.get(spec.failure_reason_output_columns[-1]))

    return "unclear"


def get_failure_reason_bucket(reason: str, spec: AnalysisSpec) -> str:
    bucket_map = spec.failure_reason_buckets or DEFAULT_FAILURE_REASON_BUCKETS
    return bucket_map.get(reason, "other")


def safe(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("._-") or "x"


def safe_pct(num: int, den: int) -> float | int:
    return round((num / den) * 100, 1) if den else 0
