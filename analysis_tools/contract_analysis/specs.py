from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_type: str
    prompt_filename: str
    required_fields: set[str]

    failure_reason_json_fields: tuple[str, ...] = ()
    failure_reason_output_columns: tuple[str, ...] = ()
    failure_test_task_keys: tuple[str, ...] = ()

    causal_json_field: str | None = None
    causal_output_column: str | None = None

    causal_display_name: str = ""
    causal_positive_label: str = ""
    causal_negative_label: str = ""

    effect_json_field: str | None = None
    effect_output_column: str | None = None
    effect_categories: tuple[str, ...] = ()
    effect_display_names: dict[str, str] | None = None

    causal_positive_effects: tuple[str, ...] = ()
    causal_negative_effects: tuple[str, ...] = ()

    failure_reason_buckets: dict[str, str] | None = None

    error_action_label: str = ""

    include_suggested_improvement: bool = False


HELPED_SPEC = AnalysisSpec(
    analysis_type="helped",
    prompt_filename="helped.txt",
    required_fields={
        "code_failure_reason",
        "contract_statements_are_correct",
        "contracts_capture_all_intended_behavior",
        "contracts_plausibly_guided_fix",
        "explanation",
    },
    failure_reason_json_fields=("code_failure_reason",),
    failure_reason_output_columns=("Vanilla Failure Reason",),
    failure_test_task_keys=("Vanilla Failing Test",),
    causal_json_field="contracts_plausibly_guided_fix",
    causal_output_column="Contract Directly Helped Fix Code",
    causal_display_name="Directly Helpful",
    causal_positive_label="Directly Helpful",
    causal_negative_label="Not Directly Helpful",
    error_action_label="Fixed",
    include_suggested_improvement=False,
)


REGRESSED_SPEC = AnalysisSpec(
    analysis_type="regressed",
    prompt_filename="regressed.txt",
    required_fields={
        "code_failure_reason",
        "contract_statements_are_correct",
        "contracts_capture_all_intended_behavior",
        "contract_effect_on_code",
        "explanation",
        "suggested_contract_improvement",
    },
    failure_reason_json_fields=("code_failure_reason",),
    failure_reason_output_columns=("Contract-Guided Failure Reason",),
    failure_test_task_keys=("Contract-Guided Failing Test",),
    causal_json_field=None,
    causal_output_column="Contract Directly Caused Bug",
    causal_display_name="Directly Harmful",
    causal_positive_label="Directly Harmful",
    causal_negative_label="Not Directly Harmful",
    effect_json_field="contract_effect_on_code",
    effect_output_column="Contract Effect on Code",
    effect_categories=(
        "no apparent contract influence",
        "added harmful validation or assertions",
        "mishandled edge or boundary cases",
        "modeled the problem incorrectly",
        "used the wrong algorithm steps",
        "formatted the output incorrectly",
        "satisfied contracts but missed task intent",
        "unclear",
    ),
    effect_display_names={
        "no apparent contract influence": "No Apparent Contract Influence",
        "added harmful validation or assertions": "Added Harmful Validation or Assertions",
        "mishandled edge or boundary cases": "Mishandled Edge or Boundary Cases",
        "modeled the problem incorrectly": "Modeled the Problem Incorrectly",
        "used the wrong algorithm steps": "Used the Wrong Algorithm Steps",
        "formatted the output incorrectly": "Formatted the Output Incorrectly",
        "satisfied contracts but missed task intent": "Satisfied Contracts but Missed Task Intent",
        "unclear": "Unclear",
    },
    causal_positive_effects=(
        "added harmful validation or assertions",
        "mishandled edge or boundary cases",
        "modeled the problem incorrectly",
        "used the wrong algorithm steps",
        "formatted the output incorrectly",
        "satisfied contracts but missed task intent",
    ),
    causal_negative_effects=(
        "no apparent contract influence",
    ),
    error_action_label="Introduced",
    include_suggested_improvement=True,
)


FAILED_SPEC = AnalysisSpec(
    analysis_type="failed",
    prompt_filename="failed.txt",
    required_fields={
        "vanilla_code_failure_reason",
        "contract_guided_code_failure_reason",
        "contract_statements_are_correct",
        "contracts_capture_all_intended_behavior",
        "contract_effect_on_code",
        "explanation",
        "suggested_contract_improvement",
    },
    failure_reason_json_fields=(
        "vanilla_code_failure_reason",
        "contract_guided_code_failure_reason",
    ),
    failure_reason_output_columns=(
        "Vanilla Failure Reason",
        "Contract-Guided Failure Reason",
    ),
    failure_test_task_keys=(
        "Vanilla Failing Test",
        "Contract-Guided Failing Test",
    ),
    causal_json_field=None,
    causal_output_column="Contract Changed Code",
    causal_display_name="Change-Inducing",
    causal_positive_label="Changed",
    causal_negative_label="No Change",
    effect_json_field="contract_effect_on_code",
    effect_output_column="Contract Effect on Code",
    effect_categories=(
        "no meaningful code change",
        "original bug unchanged",
        "original bug partially addressed",
        "original bug replaced by new failure",
        "regression dominates",
        "unclear",
    ),
    effect_display_names={
        "no meaningful code change": "No Meaningful Code Change",
        "original bug unchanged": "Original Bug Unchanged",
        "original bug partially addressed": "Original Bug Partially Addressed",
        "original bug replaced by new failure": "Original Bug Replaced by New Failure",
        "regression dominates": "Regression Dominates",
        "unclear": "Unclear",
    },
    causal_positive_effects=(
        "original bug unchanged",
        "original bug partially addressed",
        "original bug replaced by new failure",
        "regression dominates",
    ),
    causal_negative_effects=(
        "no meaningful code change",
    ),
    error_action_label="Remaining",
    include_suggested_improvement=True,
)


SPECS = {
    "helped": HELPED_SPEC,
    "regressed": REGRESSED_SPEC,
    "failed": FAILED_SPEC,
}