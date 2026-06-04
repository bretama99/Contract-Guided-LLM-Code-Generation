import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from formatting import format_analysis_summary_worksheet, format_worksheet
except ImportError:
    from analysis_tools.formatting import format_analysis_summary_worksheet, format_worksheet

from src.common.config import DATASETS


HELPED = "Helped by Contracts"
REGRESSED = "Regressed by Contracts"
FAILED = "Failed Both"

GEN_FAILS = {
    "generation_failed",
    "generation_error",
    "missing_generation",
    "empty_code",
    "empty_generated_code",
    "missing_result",
}

HELPED_COLS = [
    "Task ID",
    "Vanilla Explanation",
    "Prompt",
    "Vanilla Code",
    "Contract-Guided Code",
    "Vanilla Failing Test",
    "Preconditions",
    "Postconditions",
    "Invariants",
    "Interpretation",
]

REGRESSED_COLS = [
    "Task ID",
    "Contract-Guided Explanation",
    "Prompt",
    "Vanilla Code",
    "Contract-Guided Code",
    "Contract-Guided Failing Test",
    "Preconditions",
    "Postconditions",
    "Invariants",
    "Interpretation",
]

FAILED_COLS = [
    "Task ID",
    "Vanilla Explanation",
    "Contract-Guided Explanation",
    "Prompt",
    "Vanilla Code",
    "Contract-Guided Code",
    "Vanilla Failing Test",
    "Contract-Guided Failing Test",
    "Preconditions",
    "Postconditions",
    "Invariants",
    "Interpretation",
]

DETAIL_SHEETS = [
    (HELPED, "contract_helped", HELPED_COLS),
    (REGRESSED, "contract_regressed", REGRESSED_COLS),
    (FAILED, "both_failed", FAILED_COLS),
]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-") or "x"


def display_version(version):
    labels = {
        "raw_contracts": "Raw Contract Guided",
        "optimized_rl": "Optimized RL",
    }
    return labels.get(version, str(version).replace("_", " ").title())


def version_filename(version):
    return f"{safe(version)}_analysis.xlsx"


def pct(num, den):
    return round((num / den) * 100, 2) if den else 0


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    return re.sub(r"\s+", " ", str(value or "").replace("\r", " ")).strip()


def sorted_task_ids(task_ids):
    def key(task_id):
        return [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", str(task_id))
        ]

    return sorted(task_ids, key=key)


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------


def result_input_paths(root, dataset, provider, model, method, result_type):
    safe_provider = safe(provider)
    safe_model = safe(model)

    if DATASETS[dataset].get("evalplus_dataset"):
        return [
            root
            / "results"
            / "evalplus"
            / method
            / safe_provider
            / safe_model
            / dataset
            / f"{result_type}.json"
        ]

    if method == "vanilla":
        return [
            root
            / "results"
            / "vanilla"
            / f"stage1_{dataset}_{provider}_{safe_model}_{result_type}.json",
            root
            / "results"
            / "vanilla"
            / f"stage1_{dataset}_{safe_provider}_{safe_model}_{result_type}.json",
        ]

    return [
        root
        / "results"
        / "contract_guided_generation"
        / safe(method)
        / safe_provider
        / safe_model
        / dataset
        / f"{result_type}.json"
    ]


def first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def load_required_details(root, dataset, provider, model, method):
    path = first_existing(
        result_input_paths(root, dataset, provider, model, method, "details")
    )
    if not path:
        searched = "\n".join(
            f"  - {p}" for p in result_input_paths(root, dataset, provider, model, method, "details")
        )
        raise FileNotFoundError(
            f"{dataset}: missing details JSON file for {method!r}. Searched:\n{searched}"
        )
    return load_json(path)


def load_tasks(root, dataset):
    path = Path(DATASETS[dataset]["path"])
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return {}

    tasks = {}
    for record in load_json(path):
        task_id = str(record.get("task_id"))
        tasks[task_id] = {
            "entry_point": record.get("entry_point") or "",
            "prompt": record.get("prompt")
            or record.get("instruct_prompt")
            or record.get("complete_prompt")
            or "",
        }
    return tasks


# ---------------------------------------------------------------------------
# Result interpretation
# ---------------------------------------------------------------------------


def passed(result):
    return bool(result and result.get("passed") is True)


def ftype(result):
    result = result or {}
    evaluation = result.get("evaluation") or {}
    return str(result.get("failure_type") or evaluation.get("failure_type") or "unknown_failure")


def genfail(result):
    if not result:
        return True

    generation = result.get("generation") or {}
    return ftype(result) in GEN_FAILS or str(generation.get("status") or "") in GEN_FAILS


def code(result):
    result = result or {}
    generation = result.get("generation") or {}
    return str(
        generation.get("code")
        or generation.get("generated_code")
        or result.get("generated_code")
        or ""
    )


def result_by_task(details):
    return {
        str(result.get("task_id")): result
        for result in details
        if result.get("task_id")
    }


def all_task_ids(*detail_lists, tasks=None):
    ids = set(tasks or [])
    for details in detail_lists:
        ids.update(
            str(result.get("task_id"))
            for result in details
            if result.get("task_id")
        )
    return ids


def count_passed(details):
    return sum(1 for result in details if passed(result))


def count_failed(details):
    return sum(1 for result in details if not passed(result))


def count_generation_failures(details):
    return sum(1 for result in details if not passed(result) and genfail(result))


def prompt(tasks, task_id, result):
    if task_id in tasks:
        return tasks[task_id]["prompt"]

    task = (result or {}).get("task") or {}
    return str(
        task.get("prompt")
        or task.get("instruct_prompt")
        or task.get("complete_prompt")
        or ""
    )


def entry(result):
    result = result or {}
    return str(result.get("entry_point") or (result.get("task") or {}).get("entry_point") or "")


def explain(result):
    if not result:
        return "No result was found for this task."
    if passed(result):
        return "Passed."
    if result.get("failure_explanation"):
        return clean(result["failure_explanation"])

    evaluation = result.get("evaluation") or {}
    generation = result.get("generation") or {}
    failure_type = ftype(result)

    if genfail(result):
        return clean(
            f"Generation failed before evaluation: {generation.get('error') or failure_type}."
        )

    if failure_type == "plus_test_failure":
        stage = "plus"
    elif failure_type == "original_test_failure":
        stage = "original"
    else:
        stage = "benchmark"

    idx = evaluation.get("failed_case_index")
    inp = evaluation.get("input")
    expected = evaluation.get("expected")
    actual = evaluation.get("actual")
    error = evaluation.get("actual_error") or evaluation.get("exception")

    if expected is not None and actual is not None:
        return clean(
            f"Failed {stage} test case #{idx}. Input: {inp}. "
            f"Expected: {expected}. Actual: {actual}."
        )
    if error:
        return clean(
            f"Failed {stage} test case #{idx}. Input: {inp}. "
            f"Actual output could not be computed because: {error}"
        )
    if evaluation.get("failed_assertion"):
        return clean(f"Failed assertion: {evaluation.get('failed_assertion')}.")

    return clean(f"Failed with failure type: {failure_type}.")


def call(entry_point, inp):
    if inp is None:
        return ""

    text = str(inp).strip()
    if entry_point and re.search(rf"\b{re.escape(entry_point)}\s*\(", text):
        return text
    if entry_point and "candidate(" in text:
        return re.sub(r"\bcandidate\s*\(", f"{entry_point}(", text)
    if not entry_point:
        return text
    if text[:1] in "[(":
        return f"{entry_point}(*{text})"
    return f"{entry_point}({text})"


def failing_test(result):
    if not result or passed(result):
        return ""

    if genfail(result):
        generation = result.get("generation") or {}
        return (
            "# Generation failed before evaluation\n"
            f"# reason: {generation.get('error') or ftype(result)}"
        )

    evaluation = result.get("evaluation") or {}
    entry_point = entry(result)
    lines = []

    fields = [
        ("failure_type", ftype(result)),
        ("failed_case_index", evaluation.get("failed_case_index")),
        ("input", evaluation.get("input")),
        ("expected", evaluation.get("expected")),
        ("actual", evaluation.get("actual")),
    ]
    for key, value in fields:
        if value is not None:
            lines.append(f"# {key}: {value}")

    error = evaluation.get("actual_error") or evaluation.get("exception")
    if error:
        lines += ["# actual_error:"] + [f"# {line}" for line in str(error).splitlines()]

    lines.append("")

    if evaluation.get("failed_assertion"):
        lines.append(
            re.sub(
                r"\bcandidate\s*\(",
                f"{entry_point}(",
                str(evaluation["failed_assertion"]),
            )
        )
    else:
        rendered_call = call(entry_point, evaluation.get("input"))
        expected = evaluation.get("expected")
        if rendered_call and expected is not None:
            lines.append(f"assert {rendered_call} == {expected}")
        else:
            lines.append(rendered_call or "# Failed test input was not available.")

    return "\n".join(lines)


def bug(result):
    if not result:
        return "missing_result"
    if passed(result):
        return "passed"

    evaluation = result.get("evaluation") or {}
    text = (explain(result) + " " + str(evaluation.get("actual_error") or "")).lower()

    if genfail(result):
        return "generation failure"
    if "timeout" in text:
        return "timeout / inefficient algorithm"
    if "nameerror" in text or "not defined" in text:
        return "missing import or undefined name"
    if "syntaxerror" in text:
        return "syntax error"
    if any(token in text for token in ["typeerror", "valueerror", "traceback"]):
        return "runtime exception"
    if ftype(result) == "plus_test_failure":
        return "EvalPlus edge-case failure"
    return "wrong algorithm / wrong output"


def effect(vanilla_result, contract_result):
    if passed(vanilla_result) and passed(contract_result):
        return "both_passed"
    if not passed(vanilla_result) and passed(contract_result):
        return "contract_helped"
    if passed(vanilla_result) and not passed(contract_result):
        return "contract_regressed"
    return "both_failed"


def interpretation(effect_name, vanilla_bug, contract_bug, contract_info):
    quality = "has output guarantees only" if contract_info["Postconditions"] else "missing or weak contract"

    if effect_name == "contract_helped":
        return (
            "Contract guidance fixed a vanilla failure, likely by improving "
            f"behavior related to: {vanilla_bug}."
        )
    if effect_name == "contract_regressed":
        return (
            "Contract guidance introduced a regression. "
            f"Contract quality: {quality}. Contract-guided bug: {contract_bug}."
        )
    if effect_name == "both_failed":
        return (
            f"Both methods failed. Vanilla bug: {vanilla_bug}. "
            f"Contract-guided bug: {contract_bug}. The contract did not fix the task."
        )
    return "Both methods passed."


# ---------------------------------------------------------------------------
# Contract loading and contract-generation status
# ---------------------------------------------------------------------------


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


def contract_file_path(root, dataset, provider, model, task_id, contract_version):
    contract_dir = (
        root
        / "outputs"
        / "contracts"
        / "raw"
        / safe(provider)
        / safe(model)
        / dataset
    )

    if contract_version == "raw_contracts":
        return contract_dir / f"{safe(task_id)}_contract.json"

    if contract_version == "optimized_rl":
        return contract_dir / f"{safe(task_id)}_optimized_contract.json"

    raise ValueError(
        f"Unsupported contract_version={contract_version!r}. "
        "Add its filename pattern to contract_file_path()."
    )


def load_contract_file(root, dataset, provider, model, task_id, contract_version):
    path = contract_file_path(root, dataset, provider, model, task_id, contract_version)
    if not path.exists():
        return None
    return load_json(path)


def extract_contract_object(contract_file_data):
    if not isinstance(contract_file_data, dict):
        return {}
    contract = contract_file_data.get("contract") or {}
    return contract if isinstance(contract, dict) else {}


def get_contract(root, dataset, provider, model, task_id, contract_version="raw_contracts"):
    contract_file_data = load_contract_file(
        root=root,
        dataset=dataset,
        provider=provider,
        model=model,
        task_id=task_id,
        contract_version=contract_version,
    )
    contract = extract_contract_object(contract_file_data)

    return {
        "Preconditions": contract_text(contract.get("preconditions")),
        "Postconditions": contract_text(contract.get("postconditions")),
        "Invariants": contract_text(contract.get("invariants")),
    }


def contract_generation_status(root, dataset, provider, model, task_id, version):
    """Return the contract-generation status used for counting.

    raw_contracts are treated as generated when they appear in the result details.
    optimized_rl uses optimization_status from the optimized contract JSON file.
    """

    if version == "raw_contracts":
        return "generated"

    if version != "optimized_rl":
        raise ValueError(f"Unsupported version={version!r}")

    data = load_contract_file(
        root=root,
        dataset=dataset,
        provider=provider,
        model=model,
        task_id=task_id,
        contract_version="optimized_rl",
    )

    if not isinstance(data, dict):
        return "missing"

    return str(data.get("optimization_status") or "")


def was_attempted_contract_generation(root, dataset, provider, model, task_id, version):
    if version == "raw_contracts":
        return True

    status = contract_generation_status(root, dataset, provider, model, task_id, version)
    return status != "preserved_raw_passed"


def was_successful_contract_generation(root, dataset, provider, model, task_id, version):
    if version == "raw_contracts":
        return True

    status = contract_generation_status(root, dataset, provider, model, task_id, version)
    return status == "optimized"


# ---------------------------------------------------------------------------
# Detail rows
# ---------------------------------------------------------------------------


def detail_task_ids(root, dataset, provider, model, detail_version, detail_details):
    ids = {
        str(result.get("task_id"))
        for result in detail_details
        if result.get("task_id")
    }

    if detail_version == "raw_contracts":
        return ids

    return {
        task_id
        for task_id in ids
        if was_successful_contract_generation(
            root, dataset, provider, model, task_id, detail_version
        )
    }


def comparison_rows(
    root,
    dataset,
    provider,
    model,
    tasks,
    vanilla_details,
    contract_details,
    contract_version,
    task_ids,
):
    vanilla_by_task = result_by_task(vanilla_details)
    contract_by_task = result_by_task(contract_details)
    rows_out = []

    for task_id in sorted_task_ids(task_ids):
        vanilla_result = vanilla_by_task.get(task_id)
        contract_result = contract_by_task.get(task_id)

        contract_info = get_contract(
            root=root,
            dataset=dataset,
            provider=provider,
            model=model,
            task_id=task_id,
            contract_version=contract_version,
        )

        contract_effect = effect(vanilla_result, contract_result)

        row = {
            "Task ID": task_id,
            "Vanilla Explanation": explain(vanilla_result),
            "Contract-Guided Explanation": explain(contract_result),
            "Prompt": prompt(tasks, task_id, contract_result or vanilla_result),
            "Vanilla Code": code(vanilla_result),
            "Contract-Guided Code": code(contract_result),
            "Vanilla Failing Test": failing_test(vanilla_result),
            "Contract-Guided Failing Test": failing_test(contract_result),
            **contract_info,
        }
        row["Contract Effect"] = contract_effect
        row["Interpretation"] = interpretation(
            contract_effect,
            bug(vanilla_result),
            bug(contract_result),
            contract_info,
        )
        rows_out.append(row)

    return rows_out


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def count_successful_result_generations(details_by_task, task_ids):
    """Count tasks whose result row contains a usable generated program.

    This is intentionally different from optimized contract generation success.
    For example, an optimized_rl result may reuse a preserved raw generation; that
    still counts as a successful program generation in the cumulative summary.
    """

    return sum(
        1
        for task_id in task_ids
        if details_by_task.get(task_id) and not genfail(details_by_task[task_id])
    )


def count_passed_for_task_ids(details_by_task, task_ids):
    return sum(
        1
        for task_id in task_ids
        if passed(details_by_task.get(task_id))
    )


def count_generation_failures_for_task_ids(details_by_task, task_ids):
    """Count tasks whose code generation failed for this stage.

    This uses the same full task universe as the rest of the cumulative summary.
    A missing result row is counted as a generation failure because there is no
    usable generated program for that stage/task.
    """

    return sum(
        1
        for task_id in task_ids
        if genfail(details_by_task.get(task_id))
    )


def generation_summary_df(root, dataset, provider, model, vanilla_details, version_details):
    """Build the main cumulative summary over the full task set.

    Every row is counted against the same task universe. The optimized_rl row does
    not shrink to only tasks whose contracts were optimized; it uses every task
    present in vanilla or any contract-guided result file.
    """

    task_ids = sorted_task_ids(
        all_task_ids(vanilla_details, *(details for _, details in version_details))
    )
    total_tasks = len(task_ids)

    vanilla_by_task = result_by_task(vanilla_details)
    vanilla_passed = count_passed_for_task_ids(vanilla_by_task, task_ids)

    rows_out = [
        {
            "Stage": "Vanilla",
            "Total Tasks": total_tasks,
            "Successful Generations": count_successful_result_generations(
                vanilla_by_task, task_ids
            ),
            "Filled Generation Failures": count_generation_failures_for_task_ids(
                vanilla_by_task, task_ids
            ),
            "Cumulative Passed": vanilla_passed,
            "Cumulative Failed": total_tasks - vanilla_passed,
            "Cumulative Passed Percentage": pct(vanilla_passed, total_tasks),
        }
    ]

    for version, details in version_details:
        details_by_task = result_by_task(details)
        cumulative_passed = count_passed_for_task_ids(details_by_task, task_ids)
        generation_failures = count_generation_failures_for_task_ids(
            details_by_task,
            task_ids,
        )

        rows_out.append(
            {
                "Stage": display_version(version),
                "Total Tasks": total_tasks,
                "Successful Generations": count_successful_result_generations(
                    details_by_task, task_ids
                ),
                "Filled Generation Failures": generation_failures,
                "Cumulative Passed": cumulative_passed,
                "Cumulative Failed": total_tasks - cumulative_passed,
                "Cumulative Passed Percentage": pct(cumulative_passed, total_tasks),
            }
        )

    return pd.DataFrame(rows_out)


def rl_generation_summary_df(root, dataset, provider, model, version_details):
    rows_out = []

    for version, details in version_details:
        if version != "optimized_rl":
            continue

        attempted = 0
        successful = 0
        passed_count = 0
        failed_count = 0

        for result in details:
            task_id = str(result.get("task_id"))
            if not task_id:
                continue

            if not was_attempted_contract_generation(
                root, dataset, provider, model, task_id, version
            ):
                continue

            attempted += 1

            if was_successful_contract_generation(
                root, dataset, provider, model, task_id, version
            ):
                successful += 1

            if passed(result):
                passed_count += 1
            else:
                failed_count += 1

        rows_out.append(
            {
                "Stage": display_version(version),
                "Attempted Contract Generations": attempted,
                "Successful Contract Generations": successful,
                "Passed Task Count": passed_count,
                "Failed Task Count": failed_count,
                "Passed Task Percentage": pct(passed_count, attempted),
            }
        )

    return pd.DataFrame(rows_out)


def effect_vs_vanilla_df(vanilla_details, version_details):
    vanilla_by_task = result_by_task(vanilla_details)
    task_ids = all_task_ids(vanilla_details, *(details for _, details in version_details))
    rows_out = []

    for version, details in version_details:
        details_by_task = result_by_task(details)
        both_passed = 0
        helped = 0
        regressed = 0
        both_failed = 0

        for task_id in sorted_task_ids(task_ids):
            comparison = effect(
                vanilla_by_task.get(task_id),
                details_by_task.get(task_id),
            )

            if comparison == "both_passed":
                both_passed += 1
            elif comparison == "contract_helped":
                helped += 1
            elif comparison == "contract_regressed":
                regressed += 1
            elif comparison == "both_failed":
                both_failed += 1

        rows_out.append(
            {
                "Contract Version": display_version(version),
                "Compared Tasks": len(task_ids),
                "Both Passed": both_passed,
                "Vanilla Failed, Contract Passed": helped,
                "Contract Failed, Vanilla Passed": regressed,
                "Both Failed": both_failed,
                "Net Improvement vs Vanilla": helped - regressed,
            }
        )

    return pd.DataFrame(rows_out)


def failure_type_summary_df(vanilla_details, version_details):
    failure_types = set()

    for details in [vanilla_details, *(details for _, details in version_details)]:
        for result in details:
            if not passed(result):
                failure_types.add(ftype(result))

    latest_version = version_details[-1][0] if version_details else None
    latest_label = display_version(latest_version) if latest_version else None

    rows_out = []

    for failure_type in sorted(failure_types):
        row = {"Failure Type": failure_type}
        row["Vanilla"] = sum(
            1
            for result in vanilla_details
            if not passed(result) and ftype(result) == failure_type
        )

        for version, details in version_details:
            row[display_version(version)] = sum(
                1
                for result in details
                if not passed(result) and ftype(result) == failure_type
            )

        # Positive values mean the latest stage has fewer failures than vanilla.
        # Negative values mean the latest stage has more failures than vanilla.
        row["Net Difference"] = (
            row["Vanilla"] - row[latest_label]
            if latest_label is not None
            else 0
        )

        rows_out.append(row)

    total_row = {"Failure Type": "Total Failed"}
    total_row["Vanilla"] = count_failed(vanilla_details)

    for version, details in version_details:
        total_row[display_version(version)] = count_failed(details)

    total_row["Net Difference"] = (
        total_row["Vanilla"] - total_row[latest_label]
        if latest_label is not None
        else 0
    )

    rows_out.append(total_row)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Excel output helpers
# ---------------------------------------------------------------------------


EXCEL_CELL_LIMIT = 32767
TRUNCATION_NOTE = "\n\n[Truncated because Excel cells are limited to 32,767 characters.]"


def truncate_for_excel(value):
    """Keep cell contents within Excel/openpyxl's maximum cell length."""

    if not isinstance(value, str):
        return value

    if len(value) <= EXCEL_CELL_LIMIT:
        return value

    keep = EXCEL_CELL_LIMIT - len(TRUNCATION_NOTE)
    return value[:keep] + TRUNCATION_NOTE


def dataframe_for_excel(df):
    """Return a copy with oversized string cells truncated before to_excel()."""

    return df.map(truncate_for_excel)


# ---------------------------------------------------------------------------
# Workbook writing
# ---------------------------------------------------------------------------


def write_table(writer, df, sheet_name, title, startrow):
    """Write a table directly, without an extra title row.

    The title argument is kept so the call sites stay readable, but the Summary
    sheet now contains only tables and blank spacing between them.
    """

    dataframe_for_excel(df).to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        startrow=startrow,
    )

    # pandas writes one header row plus len(df) data rows. Add 3 to leave
    # exactly two blank rows before the next table starts.
    return startrow + len(df) + 3


def write_analysis_workbook(
    root,
    dataset,
    provider,
    model,
    detail_version,
    vanilla_details,
    version_details,
    detail_rows,
):
    output_path = (
        root
        / "analysis"
        / safe(model)
        / safe(dataset)
        / "basic"
        / version_filename(detail_version)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Summary", index=False)
        startrow = 0

        startrow = write_table(
            writer,
            generation_summary_df(
                root, dataset, provider, model, vanilla_details, version_details
            ),
            "Summary",
            "Generation / Cumulative Summary",
            startrow,
        )

        rl_summary = rl_generation_summary_df(
            root, dataset, provider, model, version_details
        )

        if not rl_summary.empty:
            startrow = write_table(
                writer,
                rl_summary,
                "Summary",
                "RL Contract Generation Summary",
                startrow,
            )

        startrow = write_table(
            writer,
            effect_vs_vanilla_df(vanilla_details, version_details),
            "Summary",
            "Effect Comparison vs Vanilla",
            startrow,
        )

        startrow = write_table(
            writer,
            failure_type_summary_df(vanilla_details, version_details),
            "Summary",
            "Failure Type Summary: Remaining Failures After Each Stage",
            startrow,
        )

        format_analysis_summary_worksheet(writer.sheets["Summary"])

        for sheet_name, effect_name, columns in DETAIL_SHEETS:
            sheet_rows = [
                row for row in detail_rows if row["Contract Effect"] == effect_name
            ]

            dataframe_for_excel(pd.DataFrame(sheet_rows, columns=columns)).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            format_worksheet(writer.sheets[sheet_name])

    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def validate_args(contract_versions, detail_version):
    if not contract_versions:
        raise ValueError("--contract-versions must include at least one version.")

    if contract_versions[0] != "raw_contracts":
        raise ValueError(
            "--contract-versions must start with raw_contracts because the "
            "generation summary treats raw contracts as the first contract-guided stage."
        )

    if detail_version not in contract_versions:
        raise ValueError(
            "--detail-version must be one of --contract-versions. "
            f"Got {detail_version!r}, expected one of {contract_versions!r}."
        )


def analyze(args):
    root = Path(args.root).resolve()
    dataset = args.dataset
    provider = args.provider
    model = args.model

    contract_versions = args.contract_versions
    detail_version = args.detail_version or contract_versions[-1]
    validate_args(contract_versions, detail_version)

    tasks = load_tasks(root, dataset)
    vanilla_details = load_required_details(root, dataset, provider, model, "vanilla")

    loaded_versions = [
        {
            "version": version,
            "details": load_required_details(root, dataset, provider, model, version),
        }
        for version in contract_versions
    ]
    version_details = [
        (item["version"], item["details"])
        for item in loaded_versions
    ]

    detail_version_data = next(
        item for item in loaded_versions if item["version"] == detail_version
    )
    detail_ids = detail_task_ids(
        root=root,
        dataset=dataset,
        provider=provider,
        model=model,
        detail_version=detail_version,
        detail_details=detail_version_data["details"],
    )

    detail_rows = comparison_rows(
        root=root,
        dataset=dataset,
        provider=provider,
        model=model,
        tasks=tasks,
        vanilla_details=vanilla_details,
        contract_details=detail_version_data["details"],
        contract_version=detail_version,
        task_ids=detail_ids,
    )

    write_analysis_workbook(
        root=root,
        dataset=dataset,
        provider=provider,
        model=model,
        detail_version=detail_version,
        vanilla_details=vanilla_details,
        version_details=version_details,
        detail_rows=detail_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--contract-versions",
        nargs="+",
        default=["raw_contracts"],
        help=(
            "Contract-guided result versions to include in summary tables, "
            "in pipeline order. Example: --contract-versions raw_contracts optimized_rl"
        ),
    )
    parser.add_argument(
        "--detail-version",
        default=None,
        help=(
            "Contract version to use for helped/regressed/failed detail sheets. "
            "Defaults to the last item in --contract-versions."
        ),
    )

    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
