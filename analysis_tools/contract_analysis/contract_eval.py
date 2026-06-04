import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_tools.formatting import format_worksheet, format_summary_worksheet
from analysis_tools.contract_analysis.common import (
    MAX_LLM_ATTEMPTS,
    SUMMARY_SHEET,
    INPUT_ERRORS_SHEET,
    OUTPUT_ERRORS_SHEET,
    OTHER_ERRORS_SHEET,
    BASE_DETAIL_COLUMNS,
    VALID_FAILURE_REASONS,
    parse_args,
    read_input_sheet,
    get_output_path,
    get_stored_analysis_path,
    read_stored_analysis_json,
    write_stored_analysis_json,
    read_prompt_template,
    get_common_prompt_parts,
    get_llm_response,
    get_task_details,
    clean_yes_no_unclear,
    clean_first_failure_reason,
    normalize_label,
    get_causal_status,
    get_bucket_failure_reason,
    get_failure_reason_bucket,
    safe_pct,
)
from analysis_tools.contract_analysis.specs import AnalysisSpec, SPECS


@dataclass
class ContractAnalysisSummary:
    analyzed: int = 0

    correct: int = 0
    complete: int = 0
    correct_and_complete: int = 0

    causal_yes: int = 0

    correct_and_causal_yes: int = 0
    correct_and_causal_no: int = 0
    not_correct_and_causal_yes: int = 0
    not_correct_and_causal_no: int = 0
    correctness_judgment_unclear: int = 0

    complete_and_causal_yes: int = 0
    complete_and_causal_no: int = 0
    not_complete_and_causal_yes: int = 0
    not_complete_and_causal_no: int = 0
    completeness_judgment_unclear: int = 0

    has_precondition: int = 0
    has_postcondition: int = 0
    has_invariant: int = 0

    input_error: int = 0
    input_error_and_has_precondition: int = 0
    input_error_and_correct: int = 0
    input_error_and_complete: int = 0
    input_error_and_causal_yes: int = 0

    output_error: int = 0
    output_error_and_has_postcondition: int = 0
    output_error_and_correct: int = 0
    output_error_and_complete: int = 0
    output_error_and_causal_yes: int = 0

    other_error: int = 0
    other_error_and_correct: int = 0
    other_error_and_complete: int = 0
    other_error_and_causal_yes: int = 0

    effect_counts: dict[str, int] = field(default_factory=dict)

    skipped: int = 0

    @property
    def total(self) -> int:
        return self.analyzed + self.skipped
    

@dataclass
class ContractAnalysisComplete:
    summary_overall: pd.DataFrame
    summary_components: pd.DataFrame
    summary_buckets: pd.DataFrame
    summary_intersections: pd.DataFrame
    summary_effects: pd.DataFrame | None
    input_errors: pd.DataFrame
    output_errors: pd.DataFrame
    other_errors: pd.DataFrame


def main() -> None:
    root, dataset, model, overwrite, analysis_type, contract_version = parse_args()
    spec = SPECS[analysis_type]
    validate_spec(spec)

    original_sheet = read_input_sheet(
        root,
        dataset,
        model,
        spec.analysis_type,
        contract_version,
    )

    result = get_contract_analysis(
        original_sheet,
        root=root,
        dataset=dataset,
        model=model,
        overwrite=overwrite,
        spec=spec,
        contract_version=contract_version,
    )

    out = get_output_path(
        root,
        dataset,
        model,
        spec.analysis_type,
        contract_version,
    )

    write_contract_analysis(out, result)

    print(f"Wrote {out}")


def validate_spec(spec: AnalysisSpec) -> None:
    if not spec.failure_reason_json_fields:
        raise ValueError(
            f"{spec.analysis_type}: at least one failure_reason_json_field is required."
        )

    if len(spec.failure_reason_json_fields) != len(spec.failure_reason_output_columns):
        raise ValueError(
            f"{spec.analysis_type}: failure_reason_json_fields and "
            "failure_reason_output_columns must have the same length."
        )

    if len(spec.failure_test_task_keys) not in {
        0,
        len(spec.failure_reason_json_fields),
    }:
        raise ValueError(
            f"{spec.analysis_type}: failure_test_task_keys should be empty or match "
            "the number of failure_reason_json_fields."
        )

    if spec.effect_output_column and not spec.effect_json_field:
        raise ValueError(
            f"{spec.analysis_type}: effect_output_column requires effect_json_field."
        )

    if spec.effect_json_field and not spec.effect_output_column:
        raise ValueError(
            f"{spec.analysis_type}: effect_json_field requires effect_output_column."
        )

    if spec.effect_output_column and spec.causal_output_column:
        if not spec.causal_positive_effects and not spec.causal_negative_effects:
            raise ValueError(
                f"{spec.analysis_type}: effect-derived causal status requires "
                "causal_positive_effects and/or causal_negative_effects."
            )

    if spec.causal_output_column and not spec.effect_output_column:
        if not spec.causal_json_field:
            raise ValueError(
                f"{spec.analysis_type}: direct causal status requires causal_json_field."
            )
        

def get_contract_analysis(
    original_sheet: pd.DataFrame,
    root: Path,
    dataset: str,
    model: str,
    overwrite: bool,
    spec: AnalysisSpec,
    contract_version: str,
) -> ContractAnalysisComplete:
    summary = ContractAnalysisSummary()
    input_errors: list[dict[str, Any]] = []
    output_errors: list[dict[str, Any]] = []
    other_errors: list[dict[str, Any]] = []

    for _, row in original_sheet.iterrows():
        task = get_task_details(row)
        log_path = get_stored_analysis_path(
            root,
            dataset,
            model,
            task_id=task["Task ID"],
            analysis_type=spec.analysis_type,
            contract_version=contract_version,
        )

        if log_path.exists() and not overwrite:
            analysis = get_stored_analysis(task, summary, log_path, spec)
        else:
            analysis = get_llm_analysis_with_retries(task, summary, log_path, spec)

        if analysis is None:
            continue

        entry = build_entry(task, analysis, spec)
        update_summary_for_task(summary, task, entry, spec)
        bucket_entry_by_failure_reason(
            entry,
            task,
            summary,
            input_errors,
            output_errors,
            other_errors,
            spec,
        )

        print(f'[DONE] {task["Task ID"]}')

    return build_contract_analysis_result(
        summary,
        input_errors,
        output_errors,
        other_errors,
        spec,
    )


def get_stored_analysis(
    task: dict[str, Any],
    summary: ContractAnalysisSummary,
    log_path: Path,
    spec: AnalysisSpec,
) -> dict[str, Any] | None:
    try:
        analysis = parse_contract_eval_response(
            read_stored_analysis_json(log_path),
            spec,
        )
    except Exception as e:
        print(
            f'[FAILED] {task["Task ID"]}: {type(e).__name__}: {e} '
            "- attempted to read, now regenerating"
        )
        traceback.print_exc()
        analysis = get_llm_analysis_with_retries(task, summary, log_path, spec)

    return analysis


def get_llm_analysis_with_retries(
    task: dict[str, Any],
    summary: ContractAnalysisSummary,
    log_path: Path,
    spec: AnalysisSpec,
) -> dict[str, Any] | None:
    for _ in range(MAX_LLM_ATTEMPTS):
        try:
            prompt = build_contract_eval_prompt(task, spec)
            response = get_llm_response(prompt)
            analysis = parse_contract_eval_response(response, spec)
            write_stored_analysis_json(log_path, analysis)
            return analysis
        except Exception as e:
            print(f'[FAILED] {task["Task ID"]}: {type(e).__name__}: {e} - retrying')

    summary.skipped += 1
    print(f'[FAILED] {task["Task ID"]} - moving to next task')
    return None


def build_contract_eval_prompt(task: dict[str, Any], spec: AnalysisSpec) -> str:
    prompt_template = read_prompt_template(spec.prompt_filename)

    prompt_values = {
        **get_common_prompt_parts(),
        "task_description": task["Prompt"],
        "vanilla_code": task["Vanilla Code"],
        "vanilla_code_failure": task.get("Vanilla Failing Test"),
        "contract_code": task["Contract-Guided Code"],
        "contract_code_failure": task.get("Contract-Guided Failing Test"),
        "contract": task["Contract"],
    }

    return prompt_template.format(**prompt_values)


def parse_contract_eval_response(
    data: dict[str, Any],
    spec: AnalysisSpec,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Parsed response must be a JSON object.")

    missing = spec.required_fields - data.keys()
    if missing:
        raise ValueError(f"Missing required field(s): {sorted(missing)}")

    explanation = data["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Field 'explanation' must be a non-empty string.")

    parsed: dict[str, Any] = {
        "contract_statements_are_correct": clean_yes_no_unclear(
            data["contract_statements_are_correct"],
            "contract_statements_are_correct",
        ),
        "contracts_capture_all_intended_behavior": clean_yes_no_unclear(
            data["contracts_capture_all_intended_behavior"],
            "contracts_capture_all_intended_behavior",
        ),
        "explanation": explanation.strip(),
    }

    for field_name in spec.failure_reason_json_fields:
        failure_reason = clean_first_failure_reason(data.get(field_name))

        if failure_reason not in VALID_FAILURE_REASONS:
            raise ValueError(
                f"Field {field_name!r} must be one of "
                f"{sorted(VALID_FAILURE_REASONS)}, got {failure_reason!r}."
            )

        parsed[field_name] = failure_reason

    if spec.causal_json_field:
        parsed[spec.causal_json_field] = clean_yes_no_unclear(
            data[spec.causal_json_field],
            spec.causal_json_field,
        )

    if spec.effect_json_field:
        effect_value = data.get(spec.effect_json_field)

        if not isinstance(effect_value, str):
            raise ValueError(f"Field {spec.effect_json_field!r} must be a string.")

        effect_value = effect_value.strip().lower()

        if spec.effect_categories and effect_value not in spec.effect_categories:
            raise ValueError(
                f"Field {spec.effect_json_field!r} must be one of "
                f"{sorted(spec.effect_categories)}, got {effect_value!r}."
            )

        parsed[spec.effect_json_field] = effect_value

    if spec.include_suggested_improvement:
        suggested_contract_improvement = data.get("suggested_contract_improvement")

        if suggested_contract_improvement is None:
            suggested_contract_improvement = ""
        elif not isinstance(suggested_contract_improvement, str):
            raise ValueError(
                "Field 'suggested_contract_improvement' must be a string or null."
            )

        suggested_contract_improvement = suggested_contract_improvement.strip()
        if not suggested_contract_improvement:
            suggested_contract_improvement = (
                "No specific contract improvement is clearly needed."
            )

        parsed["suggested_contract_improvement"] = suggested_contract_improvement

    return parsed


def get_detail_columns(spec: AnalysisSpec) -> list[str]:
    columns = [
        *BASE_DETAIL_COLUMNS,
    ]

    for failure_test_task_key in spec.failure_test_task_keys:
        columns.append(failure_test_task_key)

    for failure_reason_output_column in spec.failure_reason_output_columns:
        columns.append(failure_reason_output_column)

    columns.extend(
        [
            "Contract is Correct",
            "Contract is Complete",
        ]
    )

    if spec.causal_output_column:
        columns.append(spec.causal_output_column)

    if spec.effect_output_column:
        columns.append(spec.effect_output_column)

    columns.append("Explanation")

    if spec.include_suggested_improvement:
        columns.append("Suggested Contract Improvement")

    return columns


def build_entry(
    task: dict[str, Any],
    analysis: dict[str, Any],
    spec: AnalysisSpec,
) -> dict[str, Any]:
    entry = {
        "Task ID": task["Task ID"],
        "Prompt": task["Prompt"],
        "Vanilla Code": task["Vanilla Code"],
        "Contract-Guided Code": task["Contract-Guided Code"],
        "Contract": task["Contract"],
    }

    for failure_test_task_key in spec.failure_test_task_keys:
        entry[failure_test_task_key] = task.get(failure_test_task_key)

    for json_field, output_column in zip(
        spec.failure_reason_json_fields,
        spec.failure_reason_output_columns,
    ):
        entry[output_column] = analysis[json_field]

    entry["Contract is Correct"] = analysis["contract_statements_are_correct"]
    entry["Contract is Complete"] = analysis["contracts_capture_all_intended_behavior"]

    if spec.effect_json_field and spec.effect_output_column:
        entry[spec.effect_output_column] = analysis[spec.effect_json_field]

    if spec.causal_output_column:
        if spec.causal_json_field:
            entry[spec.causal_output_column] = analysis[spec.causal_json_field]
        else:
            entry[spec.causal_output_column] = get_causal_status(entry, spec)

    entry["Explanation"] = analysis["explanation"]

    if spec.include_suggested_improvement:
        entry["Suggested Contract Improvement"] = analysis[
            "suggested_contract_improvement"
        ]

    return entry


def update_summary_for_task(
    summary: ContractAnalysisSummary,
    task: dict[str, Any],
    entry: dict[str, Any],
    spec: AnalysisSpec,
) -> None:
    summary.analyzed += 1

    is_correct = entry["Contract is Correct"] == "yes"
    is_not_correct = entry["Contract is Correct"] == "no"

    is_complete = entry["Contract is Complete"] == "yes"
    is_not_complete = entry["Contract is Complete"] == "no"

    is_correct_and_complete = is_correct and is_complete

    causal_status = get_causal_status(entry, spec)
    is_causal_yes = causal_status == "yes"
    is_causal_no = causal_status == "no"

    if spec.effect_output_column:
        effect_value = normalize_label(entry.get(spec.effect_output_column))
        summary.effect_counts[effect_value] = (
            summary.effect_counts.get(effect_value, 0) + 1
        )
    
    if is_correct:
        summary.correct += 1

    if is_complete:
        summary.complete += 1

    if is_correct_and_complete:
        summary.correct_and_complete += 1

    if is_causal_yes:
        summary.causal_yes += 1

    if is_correct and is_causal_yes:
        summary.correct_and_causal_yes += 1
    elif is_correct and is_causal_no:
        summary.correct_and_causal_no += 1
    elif is_not_correct and is_causal_yes:
        summary.not_correct_and_causal_yes += 1
    elif is_not_correct and is_causal_no:
        summary.not_correct_and_causal_no += 1
    else:
        summary.correctness_judgment_unclear += 1

    if is_complete and is_causal_yes:
        summary.complete_and_causal_yes += 1
    elif is_complete and is_causal_no:
        summary.complete_and_causal_no += 1
    elif is_not_complete and is_causal_yes:
        summary.not_complete_and_causal_yes += 1
    elif is_not_complete and is_causal_no:
        summary.not_complete_and_causal_no += 1
    else:
        summary.completeness_judgment_unclear += 1

    if task["Has Precondition"]:
        summary.has_precondition += 1

    if task["Has Postcondition"]:
        summary.has_postcondition += 1

    if task["Has Invariant"]:
        summary.has_invariant += 1


def bucket_entry_by_failure_reason(
    entry: dict[str, Any],
    task: dict[str, Any],
    summary: ContractAnalysisSummary,
    input_errors: list[dict[str, Any]],
    output_errors: list[dict[str, Any]],
    other_errors: list[dict[str, Any]],
    spec: AnalysisSpec,
) -> None:
    error_text = get_bucket_failure_reason(entry, spec)
    is_causal_yes = get_causal_status(entry, spec) == "yes"

    is_correct = entry["Contract is Correct"] == "yes"
    is_complete = entry["Contract is Complete"] == "yes"

    bucket = get_failure_reason_bucket(error_text, spec)

    if bucket == "input":
        input_errors.append(entry)
        summary.input_error += 1

        if task["Has Precondition"]:
            summary.input_error_and_has_precondition += 1

        if is_correct:
            summary.input_error_and_correct += 1

        if is_complete:
            summary.input_error_and_complete += 1

        if is_causal_yes:
            summary.input_error_and_causal_yes += 1

    elif bucket == "output":
        output_errors.append(entry)
        summary.output_error += 1

        if task["Has Postcondition"]:
            summary.output_error_and_has_postcondition += 1

        if is_correct:
            summary.output_error_and_correct += 1

        if is_complete:
            summary.output_error_and_complete += 1

        if is_causal_yes:
            summary.output_error_and_causal_yes += 1

    else:
        other_errors.append(entry)
        summary.other_error += 1

        if is_correct:
            summary.other_error_and_correct += 1

        if is_complete:
            summary.other_error_and_complete += 1

        if is_causal_yes:
            summary.other_error_and_causal_yes += 1


def build_contract_analysis_result(
    summary: ContractAnalysisSummary,
    input_errors: list[dict[str, Any]],
    output_errors: list[dict[str, Any]],
    other_errors: list[dict[str, Any]],
    spec: AnalysisSpec,
) -> ContractAnalysisComplete:
    detail_columns = get_detail_columns(spec)

    return ContractAnalysisComplete(
        summary_overall=build_overall_summary_df(summary, spec),
        summary_components=build_component_summary_df(summary),
        summary_buckets=build_bucket_summary_df(summary, spec),
        summary_intersections=build_intersection_summary_df(summary, spec),
        summary_effects=build_effect_summary_df(summary, spec),
        input_errors=pd.DataFrame(input_errors, columns=detail_columns),
        output_errors=pd.DataFrame(output_errors, columns=detail_columns),
        other_errors=pd.DataFrame(other_errors, columns=detail_columns),
    )


def build_overall_summary_df(
    summary: ContractAnalysisSummary,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    rows = [
        [summary.total, ""],
        [summary.analyzed, safe_pct(summary.analyzed, summary.total)],
        [summary.skipped, safe_pct(summary.skipped, summary.total)],
        [summary.correct, safe_pct(summary.correct, summary.analyzed)],
        [summary.complete, safe_pct(summary.complete, summary.analyzed)],
        [
            summary.correct_and_complete,
            safe_pct(summary.correct_and_complete, summary.analyzed),
        ],
        [summary.causal_yes, safe_pct(summary.causal_yes, summary.analyzed)],
    ]

    return pd.DataFrame(
        rows,
        columns=["Count", "Percentage"],
        index=[
            "Total Tasks",
            "Analyzed Tasks",
            "Skipped Tasks",
            "Correct Contracts",
            "Complete Contracts",
            "Correct and Complete Contracts",
            f"{spec.causal_display_name} Contracts",
        ],
    )


def build_component_summary_df(
    summary: ContractAnalysisSummary,
) -> pd.DataFrame:
    rows = [
        [summary.has_precondition, safe_pct(summary.has_precondition, summary.analyzed)],
        [summary.has_postcondition, safe_pct(summary.has_postcondition, summary.analyzed)],
        [summary.has_invariant, safe_pct(summary.has_invariant, summary.analyzed)],
    ]

    return pd.DataFrame(
        rows,
        columns=["Count", "Percentage"],
        index=[
            "Contracts with Preconditions",
            "Contracts with Postconditions",
            "Contracts with Invariants",
        ],
    )


def build_intersection_summary_df(
    summary: ContractAnalysisSummary,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    rows = [
        [
            summary.correct_and_causal_yes,
            summary.correct_and_causal_no,
            summary.not_correct_and_causal_yes,
            summary.not_correct_and_causal_no,
            summary.correctness_judgment_unclear,
        ],
        [
            summary.complete_and_causal_yes,
            summary.complete_and_causal_no,
            summary.not_complete_and_causal_yes,
            summary.not_complete_and_causal_no,
            summary.completeness_judgment_unclear,
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            f"Yes + {spec.causal_positive_label}",
            f"Yes + {spec.causal_negative_label}",
            f"No + {spec.causal_positive_label}",
            f"No + {spec.causal_negative_label}",
            "Unclear",
        ],
        index=[
            "Correct",
            "Complete",
        ],
    )


def build_bucket_summary_df(
    summary: ContractAnalysisSummary,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    rows = [
        [
            summary.input_error,
            safe_pct(summary.input_error_and_correct, summary.input_error),
            safe_pct(summary.input_error_and_complete, summary.input_error),
            safe_pct(summary.input_error_and_causal_yes, summary.input_error),
            safe_pct(summary.input_error_and_has_precondition, summary.input_error),
        ],
        [
            summary.output_error,
            safe_pct(summary.output_error_and_correct, summary.output_error),
            safe_pct(summary.output_error_and_complete, summary.output_error),
            safe_pct(summary.output_error_and_causal_yes, summary.output_error),
            safe_pct(summary.output_error_and_has_postcondition, summary.output_error),
        ],
        [
            summary.other_error,
            safe_pct(summary.other_error_and_correct, summary.other_error),
            safe_pct(summary.other_error_and_complete, summary.other_error),
            safe_pct(summary.other_error_and_causal_yes, summary.other_error),
            "",
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Number of Tasks",
            "% with Correct Contracts",
            "% with Complete Contracts",
            f"% with {spec.causal_display_name} Contracts",
            "% with Relevant Contract Type",
        ],
        index=[
            "Input-Related Errors",
            "Output-Related Errors",
            "Other Errors",
        ],
    )


def build_effect_summary_df(
    summary: ContractAnalysisSummary,
    spec: AnalysisSpec,
) -> pd.DataFrame | None:
    if not spec.effect_output_column:
        return None

    categories = spec.effect_categories or tuple(sorted(summary.effect_counts))

    if not categories:
        return None

    display_names = spec.effect_display_names or {}

    rows = []
    index = []

    for category in categories:
        count = summary.effect_counts.get(category, 0)
        rows.append([count, safe_pct(count, summary.analyzed)])
        index.append(display_names.get(category, category))

    return pd.DataFrame(
        rows,
        columns=["Count", "Percentage"],
        index=index,
    )


def write_contract_analysis(
    out: Path,
    result: ContractAnalysisComplete,
) -> None:
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        result.summary_overall.to_excel(
            writer,
            sheet_name=SUMMARY_SHEET,
            index=True,
            startrow=0,
            startcol=0,
        )

        result.summary_components.to_excel(
            writer,
            sheet_name=SUMMARY_SHEET,
            index=True,
            header=False,
            startrow=9,
            startcol=0,
        )

        if result.summary_effects is not None:
            result.summary_effects.to_excel(
                writer,
                sheet_name=SUMMARY_SHEET,
                index=True,
                header=False,
                startrow=13,
                startcol=0,
            )

        result.summary_intersections.to_excel(
            writer,
            sheet_name=SUMMARY_SHEET,
            index=True,
            startrow=0,
            startcol=4,
        )

        result.summary_buckets.to_excel(
            writer,
            sheet_name=SUMMARY_SHEET,
            index=True,
            startrow=5,
            startcol=4,
        )

        result.input_errors.to_excel(
            writer,
            sheet_name=INPUT_ERRORS_SHEET,
            index=False,
        )
        result.output_errors.to_excel(
            writer,
            sheet_name=OUTPUT_ERRORS_SHEET,
            index=False,
        )
        result.other_errors.to_excel(
            writer,
            sheet_name=OTHER_ERRORS_SHEET,
            index=False,
        )

        format_summary_worksheet(writer.sheets[SUMMARY_SHEET])
        format_worksheet(writer.sheets[INPUT_ERRORS_SHEET])
        format_worksheet(writer.sheets[OUTPUT_ERRORS_SHEET])
        format_worksheet(writer.sheets[OTHER_ERRORS_SHEET])


if __name__ == "__main__":
    main()