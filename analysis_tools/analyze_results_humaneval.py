import argparse
from pathlib import Path
from typing import Any
import re
import json
import pandas as pd

from formatting import format_worksheet
from src.common.config import DATASETS
from src.common.raw_contract_paths import raw_contract_path

# Path variables
ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = DATASETS["evalplus"]["path"]
OUTPUT = ROOT / "analysis" / "gpt-3.5-turbo" / "evalplus_analysis.xlsx"
RESULTS_VANILLA_PATH = ROOT / "results" / "evalplus" / "vanilla" / "openrouter" / "openai_gpt-3.5-turbo" / "evalplus"
RESULTS_CONTRACT_PATH = ROOT / "results" / "evalplus" / "raw_contracts" / "openrouter" / "openai_gpt-3.5-turbo" / "evalplus"


def main():
    global task_info
    task_info = extract_task_info()

    get_summary()
    analyze_contracts()

    print("Data written successfully.")
    
    
def get_summary():

    # get summary file name
    vanilla_summary_file = RESULTS_VANILLA_PATH / "summary.json"
    contract_summary_file = RESULTS_CONTRACT_PATH / "summary.json"
    
    # get data inside files
    with open(vanilla_summary_file, "r", encoding="utf-8") as v_file, open(contract_summary_file, "r", encoding="utf-8") as c_file:
        vdata = json.load(v_file)
        cdata = json.load(c_file)
    
    # row: vanilla
    row_vanilla = {
        "Total Tasks": vdata.get("total_tasks"),
        "Generation Successful": vdata.get("successful_generation_count"),
        "Passed": vdata.get("passed"),
        "Failed": vdata.get("failed"),
        "Pass Percentage": vdata.get("pass@1_percent")
    }

    # row: contracts
    row_contracts = {
        "Total Tasks": cdata.get("total_tasks"),
        "Generation Successful": cdata.get("successful_generation_count"),
        "Passed": cdata.get("passed"),
        "Failed": cdata.get("failed"),
        "Pass Percentage": cdata.get("pass@1_percent")
    }

    # convert to dataframe
    df = pd.DataFrame(
        [row_vanilla, row_contracts],
        index=["Vanilla", "Contracts"]
    ).T

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=True)
        format_worksheet(writer.sheets["Summary"])


def analyze_contracts():

    # get summary file name
    vanilla_detail_file = RESULTS_VANILLA_PATH / "details.json"
    contract_detail_file = RESULTS_CONTRACT_PATH / "details.json"
    
    # get data inside files
    with open(vanilla_detail_file, "r", encoding="utf-8") as v_file, open(contract_detail_file, "r", encoding="utf-8") as c_file:
        vdata = json.load(v_file)
        cdata = json.load(c_file)

    vanilla_info = {}
    contract_info = {}

    # extract important details
    for item in vdata:
        vanilla_info[item.get("task_id")] = {
            "passed": item.get("passed"),
            "explanation": item.get("failure_explanation"),
            "code": item.get("generation", {}).get("code")
        }

    for item in cdata:
        contract_info[item.get("task_id")] = {
            "passed": item.get("passed"),
            "explanation": item.get("failure_explanation"),
            "code": item.get("generation", {}).get("code"),
        }

    # compare pass/fail results for vanilla and contract code and add to appropriate list
    all_results = []
    fail_fail = []
    fail_pass = []
    pass_fail = []

    for task in task_info:

        contract = extract_contract(task)
        entry = get_entry(task, contract, vanilla_info, contract_info)
        all_results.append(entry)

        v_fail = not vanilla_info[task]["passed"]
        c_fail = not contract_info[task]["passed"]

        if v_fail and c_fail:
            fail_fail.append(entry)
        elif v_fail and not c_fail:
            fail_pass.append(entry)
        elif not v_fail and c_fail:
            pass_fail.append(entry)
    
    # export each df to a separate sheet
    sheets = {
        "All Tasks": pd.DataFrame(all_results),
        "Failed Both": pd.DataFrame(fail_fail),
        "Failed Vanilla Only": pd.DataFrame(fail_pass),
        "Failed Contracts Only": pd.DataFrame(pass_fail)
    }

    with pd.ExcelWriter(OUTPUT, engine="openpyxl", mode="a") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            format_worksheet(writer.sheets[sheet_name])
          

def get_entry(task, contract, vanilla_info, contract_info):
    
    vanilla_status = "PASS" if vanilla_info[task]["passed"] else "FAIL"
    contract_status = "PASS" if contract_info[task]["passed"] else "FAIL"

    return {
        "Task ID": task,
        "Entry Point": task_info[task]["entry_point"],
        "Vanilla Status": vanilla_status,
        "Contract-Guided Status": contract_status,
        "Vanilla Explanation": vanilla_info[task]["explanation"],
        "Contract-Guided Explanation": contract_info[task]["explanation"],
        "Prompt": task_info[task]["prompt"],
        "Vanilla Code": vanilla_info[task]["code"],
        "Contract-Guided Code": contract_info[task]["code"],
        "Preconditions": contract["preconditions"],
        "Postconditions": contract["postconditions"],
        "Invariants": contract["invariants"],
    }


def extract_task_info():

    with open(DATASET_FILE, "r", encoding="utf-8") as d:
        dataset = json.load(d)

    tasks = {}
    
    for item in dataset:
        tasks[item.get("task_id")] = {
            "entry_point": item.get("entry_point"),
            "prompt": item.get("prompt")
        }

    return tasks


def extract_contract(task):
    
    contract_file = raw_contract_path("evalplus", task, "openrouter", "openai_gpt-3.5-turbo")
    
    with open(contract_file, "r", encoding="utf-8") as c:
        contract_data = json.load(c)
    
    contract = contract_data.get("contract", {})
    preconditions = "\n".join([item["description"] for item in contract.get("preconditions")])
    postconditions = "\n".join([item["description"] for item in contract.get("postconditions")])
    invariants = "\n".join([item["description"] for item in contract.get("invariants")])


    return {
        "task": task,
        "preconditions": preconditions, 
        "postconditions": postconditions, 
        "invariants": invariants
    }


if __name__ == "__main__":
    main()