from pathlib import Path
import pandas as pd
from analysis_tools.general_analysis.analyze_all import safe
from analysis_tools.formatting import format_worksheet


"""
IMPORTANT: Sorting is currently hard-coded, need to edit later
"""

RAW_HELPED = "Vanilla Failed, Raw Passed"
RAW_REGRESSED = "Vanilla Passed, Raw Failed"
RAW_FAILED = "Vanilla Failed, Raw Failed"
OPTIMIZED_HELPED = "Raw Failed, Optimized Helped"
OPTIMIZED_FAILED = "Raw Failed, Optimized Failed"


def write_sorted_tasks_file(sorted_tasks: dict[str, list], root: Path, dataset: str, model: str) -> None:

    out = get_output_path(root, dataset, model)

    with pd.ExcelWriter(out, engine="openpyxl") as writer: 
        for name, rows in sorted_tasks.items():
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=name, index=False)
            format_worksheet(writer.sheets[name])

    print(f"Wrote {out}")


def get_output_path(root: Path, dataset: str, model: str) -> Path:

    safe_model = safe(model)
    safe_dataset = safe(dataset)

    out = root / "analysis" / safe_model / safe_dataset / f"{safe_model}_{safe_dataset}_sorted.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def sort_tasks(task_rows: list[dict]) -> dict[str, list]:
    
    result = {
        RAW_HELPED: [],
        RAW_REGRESSED: [],
        RAW_FAILED: [],
        OPTIMIZED_HELPED: [],
        OPTIMIZED_FAILED: [],
    }
    
    for task in task_rows:

        vanilla_passed = task["Vanilla Passed"]
        raw_passed = task["Raw Contract-Guided Passed"]
        optimized_passed = task["Optimized Contract-Guided Passed"]

        if not vanilla_passed and raw_passed:
            result[RAW_HELPED].append(task)
        elif vanilla_passed and not raw_passed:
            result[RAW_REGRESSED].append(task)
        elif not vanilla_passed and not raw_passed:
            result[RAW_FAILED].append(task)

        if not raw_passed and optimized_passed:
            result[OPTIMIZED_HELPED].append(task)
        elif not raw_passed and not optimized_passed:
            result[OPTIMIZED_FAILED].append(task)

    return result
