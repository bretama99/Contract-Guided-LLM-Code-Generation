import argparse
import json
import re
from pathlib import Path
from typing import Any
import pandas as pd
from dataclasses import dataclass
from analysis_tools.formatting import format_analysis_summary_worksheet, format_worksheet
from src.common.config import DATASETS
from sort_results import sort_tasks, write_sorted_tasks_file
from summarize_results import construct_summary_sheet


NOT_APPLICABLE = "n/a"

SUPPORTED_CONTRACT_VERSIONS = ["raw_contracts", "optimized_rl"]

CONTRACT_VERSION_DISPLAY_NAME = {
    "raw_contracts": "Raw Contract",
    "optimized_rl": "Optimized Contract",
}

CONTRACT_VERSION_GUIDED_NAME = {
    "raw_contracts": "Raw Contract-Guided",
    "optimized_rl": "Optimized Contract-Guided",
}

TASK_COLUMNS_VANILLA = [
    "Task ID", 
    "Prompt", 
    "Vanilla Code", 
    "Vanilla Passed", 
    "Vanilla Failing Test"
]

TASK_COLUMNS_CONTRACT = [
    "Contract Generation Attempted", 
    "Contract Generation Succeeded",
    "Contract Preconditions",
    "Contract Postconditions",
    "Contract Invariants",
    "Contract-Guided Code",
    "Contract-Guided Passed",
    "Contract-Guided Failing Test"
]

# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def main():
    
    root, datasets, provider, model, contract_versions = parse_args()

    summary_sheets = {}
    detail_sheets = {}

    for dataset in datasets:

        task_ids = get_task_id_list(dataset)

        vanilla_evaluation_details = get_evaluation_details(root, dataset, provider, model, "vanilla")

        contracts_evaluation_details = {}
        contracts_unaltered = {}
        for contract_version in contract_versions:
            contracts_evaluation_details[contract_version] = get_evaluation_details(root, dataset, provider, model, contract_version)
            contracts_unaltered[contract_version] = get_all_contracts(root, dataset, provider, model, contract_version, task_ids)

        # convert the lists of jsons into dictionaries with column info
        task_rows = get_task_rows(task_ids, vanilla_evaluation_details, contracts_evaluation_details, contracts_unaltered)

        # sorts tasks into different lists depending on pass/fail data (for future in-depth analysis)
        sorted_tasks = sort_tasks(task_rows)
        write_sorted_tasks_file(sorted_tasks, root, dataset, model)

        # construct summary sheet and detail sheet for this dataset
        #summary_sheets[dataset] = construct_summary_sheet(task_rows, sorted_tasks)
        detail_sheets[dataset] = pd.DataFrame(task_rows)

    # get output paths
    #summary_out = get_output_path(root, model, "summary")
    details_out = get_output_path(root, model, "details")

    # then write the summary
    #write_excel_file(summary_sheets, summary_out)
    write_excel_file(detail_sheets, details_out)
    print("Analysis Complete")


# ---------------------------------------------------------------------------
# Write Excel
# ---------------------------------------------------------------------------

def write_excel_file(sheets: dict[str, pd.DataFrame], out: Path) -> None:
    with pd.ExcelWriter(out, engine="openpyxl") as writer: 
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            format_worksheet(writer.sheets[name])


def get_output_path(root: Path, model: str, type: str) -> Path:
    safe_model = safe(model)
    out = root / "analysis" / safe_model / f"{safe_model}_{type}_analysis.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

        
# ---------------------------------------------------------------------------
# Get Info for ALL TASKS in a dataset for a specific CONTRACT VERSION
# ---------------------------------------------------------------------------

# returns a dictionary of evaluation details for each task
def get_evaluation_details(root: Path, dataset: str, provider: str, model: str, contract_version: str) -> dict[str, dict]:
    
    dir = root / "results"

    if DATASETS[dataset].get("evalplus_dataset"):
        path = dir / "evalplus" / contract_version / safe(provider) / safe(model) / dataset / "details.json"
    else:
        raise ValueError("DATASET SUPPORT NOT IMPLEMENTED YET. PLEASE IMPLEMENT NOW")

    if not path.exists():
        raise FileNotFoundError(f"Missing details json: {path}")
    
    return {
        record["task_id"]: record
        for record in read_json(path)
    }



# returns a dictionary of all original contracts for a particular stage (contract object not extracted, it returns the raw info)
def get_all_contracts(root: Path, dataset: str, provider: str, model: str, contract_version: str, task_ids: list[str]) -> dict[str, dict]:
    return {
        task_id: read_contract_json(root, dataset, provider, model, contract_version, task_id)
        for task_id in task_ids
    }

    
# ---------------------------------------------------------------------------
# Get Info Helpers
# ---------------------------------------------------------------------------

# returns single contract json as a dict
def read_contract_json(root: Path, dataset: str, provider: str, model: str, contract_version: str, task_id: str) -> dict:
    
    dir = root / "contracts" / "raw" / safe(provider) / safe(model) / dataset

    if contract_version == "raw_contracts":
        path = dir / f"{safe(task_id)}_contract.json"
    elif contract_version == "optimized_rl":
        path = dir / f"{safe(task_id)}_optimized_contract.json"
    else:
        raise ValueError(f"Could not read contract. Unsupported contract_version={contract_version!r}.")
    
    if not path.exists():
        raise FileNotFoundError(f"Missing contract json: {path}")
    
    return read_json(path)
    

# return list of task id names (not safe for file name usage)
def get_task_id_list(dataset: str) -> list[str]:
    path = Path(DATASETS[dataset]["path"])
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
    
    return [str(record["task_id"]) for record in read_json(path)]


# ---------------------------------------------------------------------------
# Get Task Rows
# ---------------------------------------------------------------------------

# returns a list of all task rows, which can be used to construct the full task summary sheet later
def get_task_rows(task_ids: list[str], vanilla_eval_details: dict[str, dict], contracts_eval_details: dict[str, dict], contracts_unaltered: dict[str, dict]) -> list[dict]:
    result = []

    for task_id in task_ids:
        # begin a new row with the vanilla evaluation
        row = extract_vanilla_row(vanilla_eval_details[task_id])
        # append to the row for each contract version
        for contract_version in contracts_eval_details.keys():
            row.update(extract_contract_row(
                contract_version,
                contracts_eval_details[contract_version][task_id],
                contracts_unaltered[contract_version][task_id],
            ))
        result.append(row)

    return result


# get the vanilla code details for the specified task
def extract_vanilla_row(task_evaluation: dict) -> dict:
    vanilla_passed = task_evaluation["evaluation"]["status"] == "passed"
    vanilla_failing_test = get_failing_test_string(task_evaluation)

    return {
        "Task ID": task_evaluation["task_id"],
        "Prompt": task_evaluation["task"]["prompt"],
        "Vanilla Code": task_evaluation["generation"]["code"],
        "Vanilla Passed": vanilla_passed,
        "Vanilla Failing Test": vanilla_failing_test if not vanilla_passed else NOT_APPLICABLE,
    }


# get the contract and contract-guided code details for the specified task
def extract_contract_row(contract_version: str, task_evaluation: dict, contract_raw: dict):
    contract_version_name = CONTRACT_VERSION_DISPLAY_NAME[contract_version]
    contract_version_guided_name = CONTRACT_VERSION_GUIDED_NAME[contract_version]
    contract_generation_attempted = get_contract_generation_attempted(contract_raw)
    contract_generation_succeeded = get_contract_generation_succeeded(contract_raw)
    contract = extract_contract(contract_raw)
    contract_version_passed = task_evaluation["evaluation"]["status"] == "passed"
    contract_failing_test = get_failing_test_string(task_evaluation)

    return {
        f"{contract_version_name} Generation Attempted": contract_generation_attempted,
        f"{contract_version_name} Generation Succeeded": contract_generation_succeeded,
        f"{contract_version_name} Preconditions": contract["Preconditions"],
        f"{contract_version_name} Postconditions": contract["Postconditions"],
        f"{contract_version_name} Invariants": contract["Invariants"],
        f"{contract_version_guided_name} Code": task_evaluation["generation"]["code"],
        f"{contract_version_guided_name} Passed": contract_version_passed,
        f"{contract_version_guided_name} Failing Test": contract_failing_test if not contract_version_passed else NOT_APPLICABLE,
    }


# ---------------------------------------------------------------------------
# Get Task Row Helpers
# ---------------------------------------------------------------------------

# get the failure info as a string
def get_failing_test_string(task_evaluation: dict) -> str:
    return "\n".join([
        "failure type: " + task_evaluation["evaluation"]["failure type"] or NOT_APPLICABLE,
        "failed input: " + task_evaluation["evaluation"]["input"] or NOT_APPLICABLE,
        "expected: " + task_evaluation["evaluation"]["expected"] or NOT_APPLICABLE,
        "actual: " + task_evaluation["evaluation"]["actual"] or NOT_APPLICABLE,
        "exception: " + task_evaluation["evaluation"]["exception"] or NOT_APPLICABLE,
    ])


# returns true if the contract generation was attempted, otherwise false
def get_contract_generation_attempted(contract: dict) -> bool:
    
    if "optimization_status" not in contract:
        return True
    
    return not contract["optimization_status"] == "preserved_raw_passed"


# returns true if the contract generation succeeded, otherwise false
def get_contract_generation_succeeded(contract: dict) -> bool:

    if get_contract_generation_attempted(contract):
        return contract["status"] == "success"
    
    return None


# extracts the preconditions, postconditions, and invariants
def extract_contract(contract_raw: dict) -> dict:
    
    contract = contract_raw["contract"]
    return {
        "Preconditions": contract_text(contract.get("preconditions")),
        "Postconditions": contract_text(contract.get("postconditions")),
        "Invariants": contract_text(contract.get("invariants")),
    }


# parses contract text
def contract_text(value):
    if not value:
        return ""

    if isinstance(value, list):
        return "\n".join(filter(None, map(contract_text, value)))

    if isinstance(value, dict):
        for key in ["description", "condition", "expression", "expected_behavior", "case"]:
            if value.get(key):
                return str(value[key]).strip()
        return "\n".join(filter(None, (contract_text(v) for v in value.values())))

    return str(value).strip()


# ---------------------------------------------------------------------------
# Parse Command Line Arguments
# ---------------------------------------------------------------------------

def parse_args() -> tuple[Path, tuple[str], str, str, tuple[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS) + ["all"], required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract-versions", nargs="+", choices=SUPPORTED_CONTRACT_VERSIONS, required=True)
    args = parser.parse_args()

    if "all" in args.datasets:
        selected_datasets = sorted(DATASETS)
    else:
        selected_datasets = args.datasets

    return (
        Path(args.root).resolve(),
        selected_datasets,
        args.provider,
        args.model,
        args.contract_versions,
    )


# ---------------------------------------------------------------------------
# General Helpers
# ---------------------------------------------------------------------------

def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-") or "x"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()