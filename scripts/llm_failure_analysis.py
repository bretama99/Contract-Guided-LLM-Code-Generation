from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client


JsonDict = dict[str, Any]


CONTRACT_TYPES = [
    "interface/signature",
    "constructor/state initialization",
    "precondition",
    "postcondition/return value",
    "postcondition/state effect",
    "invariant",
    "edge case",
    "invalid input behavior",
    "dependency/interaction",
    "syntax/formatting",
    "implementation completeness",
    "not contract-related",
]


def safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    cleaned = cleaned.strip("._-")
    return cleaned or "unnamed"


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def short(value: Any, limit: int = 5000) -> str:
    content = text(value)
    return content if len(content) <= limit else content[:limit] + "\n...[truncated]"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(value: Any, limit: int = 6000) -> str:
    try:
        content = json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        content = str(value)
    return short(content, limit)


def task_identifier(task: JsonDict) -> str:
    for key in ("task_id", "id", "name", "entry_point"):
        value = text(task.get(key))
        if value:
            return value
    raise ValueError("Dataset task is missing an id/task_id/name/entry_point.")


def find_dataset_file(root: Path, benchmark: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else root / path

    candidates = [
        root / "data" / "processed" / benchmark / "ClassEval_data.json",
        root / "data" / "processed" / benchmark / f"{benchmark}_data.json",
        root / "data" / "processed" / benchmark / f"{benchmark}.json",
        root / "data" / benchmark / f"{benchmark}.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    found = sorted((root / "data").rglob("*.json")) if (root / "data").exists() else []
    for path in found:
        if benchmark.lower() in str(path).lower():
            return path

    raise FileNotFoundError(
        f"Could not find dataset file for benchmark={benchmark}. "
        "Pass --dataset-file explicitly."
    )


def load_dataset_by_task(root: Path, benchmark: str, explicit: str | None) -> dict[str, JsonDict]:
    path = find_dataset_file(root, benchmark, explicit)
    data = load_json(path)

    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            rows = data["tasks"]
        elif "data" in data and isinstance(data["data"], list):
            rows = data["data"]
        else:
            rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    result: dict[str, JsonDict] = {}
    for item in rows:
        if isinstance(item, dict):
            result[task_identifier(item)] = item

    return result


def remove_solution_fields(value: Any, *, include_test_source: bool, test_source_limit: int) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            lower = key.lower()

            if "solution" in lower or lower in {"canonical_solution", "reference_solution"}:
                continue

            if not include_test_source and (
                lower == "test"
                or lower.endswith("_test")
                or "test_code" in lower
                or "method_test" in lower
            ):
                continue

            if include_test_source and lower == "test":
                cleaned[key] = short(item, test_source_limit)
                continue

            cleaned[key] = remove_solution_fields(
                item,
                include_test_source=include_test_source,
                test_source_limit=test_source_limit,
            )
        return cleaned

    if isinstance(value, list):
        return [
            remove_solution_fields(
                item,
                include_test_source=include_test_source,
                test_source_limit=test_source_limit,
            )
            for item in value
        ]

    return value


def find_details_file(
    root: Path,
    benchmark: str,
    pipeline: str,
    provider: str,
    model: str,
    explicit: str | None,
) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else root / path

    safe_provider = safe_name(provider)
    safe_model = safe_name(model)

    search_root = root / "results" / benchmark / pipeline
    candidates = sorted(search_root.rglob("*details.json")) if search_root.exists() else []
    filtered = [
        path
        for path in candidates
        if safe_provider in path.name and safe_model in path.name
    ]
    if filtered:
        return filtered[-1]

    all_candidates = sorted((root / "results").rglob("*details.json"))
    filtered = [
        path
        for path in all_candidates
        if benchmark in str(path)
        and pipeline in str(path)
        and safe_provider in path.name
        and safe_model in path.name
    ]
    if filtered:
        return filtered[-1]

    raise FileNotFoundError(
        f"Could not find {pipeline} details for {provider}/{model}. "
        f"Searched under {root / 'results'}."
    )


def resolve_project_path(root: Path, raw_path: Any) -> Path | None:
    value = text(raw_path)
    if not value:
        return None

    path = Path(value)
    if path.exists():
        return path

    parts = path.parts
    for anchor in ("outputs", "results"):
        if anchor in parts:
            idx = parts.index(anchor)
            candidate = root.joinpath(*parts[idx:])
            if candidate.exists():
                return candidate

    candidate = root / value
    if candidate.exists():
        return candidate

    return None


def load_generation_record(root: Path, detail: JsonDict) -> JsonDict:
    generation = as_dict(detail.get("generation"))
    path = resolve_project_path(root, generation.get("path"))
    if path and path.exists():
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    return {}


def contract_file_for_task(
    root: Path,
    benchmark: str,
    provider: str,
    model: str,
    task_id: str,
) -> Path:
    return (
        root
        / "outputs"
        / benchmark
        / "contracts"
        / safe_name(provider)
        / safe_name(model)
        / f"{safe_name(task_id)}_contract.json"
    )


def load_contract(
    root: Path,
    benchmark: str,
    provider: str,
    model: str,
    task_id: str,
    generation_record: JsonDict,
) -> JsonDict:
    contract = generation_record.get("contract")
    if isinstance(contract, dict):
        return contract

    path = contract_file_for_task(root, benchmark, provider, model, task_id)
    if not path.exists():
        return {}

    record = load_json(path)
    if isinstance(record, dict):
        return as_dict(record.get("contract"))
    return {}


def parsed_failure_blocks(stderr_or_error: str) -> list[JsonDict]:
    output = text(stderr_or_error)
    blocks: list[JsonDict] = []

    pattern = re.compile(
        r"=+\n"
        r"(FAIL|ERROR):\s+(test[^\s]*)\s+\(([^)]+)\)\n"
        r"-+\n"
        r"(.*?)(?=\n=+\n|\n-+\nRan\s+\d+\s+tests?|\Z)",
        re.DOTALL,
    )

    for match in pattern.finditer(output):
        status = match.group(1)
        test_name = match.group(2)
        dotted = match.group(3)
        traceback = match.group(4).strip()
        test_class = dotted.split(".")[-2] if "." in dotted else dotted

        expected, actual = extract_expected_actual(traceback)

        blocks.append(
            {
                "status": status,
                "test_class": test_class,
                "test_name": test_name,
                "traceback": short(traceback, 2500),
                "assertion_or_error": short(last_interesting_line(traceback), 1000),
                "expected_result": short(expected, 1000),
                "actual_result": short(actual, 1000),
            }
        )

    if blocks:
        return blocks

    summary_pattern = re.compile(
        r"^(test[^\s]*)\s+\(([^)]+)\)\s+\.\.\.\s+(FAIL|ERROR)$",
        re.MULTILINE,
    )
    for match in summary_pattern.finditer(output):
        dotted = match.group(2)
        test_class = dotted.split(".")[-2] if "." in dotted else dotted
        blocks.append(
            {
                "status": match.group(3),
                "test_class": test_class,
                "test_name": match.group(1),
                "traceback": "",
                "assertion_or_error": "",
                "expected_result": "",
                "actual_result": "",
            }
        )

    return blocks


def last_interesting_line(traceback: str) -> str:
    lines = [line.strip() for line in traceback.splitlines() if line.strip()]
    if not lines:
        return ""

    markers = (
        "AssertionError",
        "TypeError",
        "ValueError",
        "KeyError",
        "AttributeError",
        "IndexError",
        "NameError",
        "SyntaxError",
        "OperationalError",
    )
    for line in reversed(lines):
        if any(marker in line for marker in markers):
            return line
    return lines[-1]


def extract_expected_actual(traceback: str) -> tuple[str, str]:
    lines = [line.rstrip() for line in traceback.splitlines()]

    for line in lines:
        stripped = line.strip()

        match = re.search(r"AssertionError:\s*(.+?)\s*!=\s*(.+)$", stripped)
        if match:
            return match.group(1), match.group(2)

        match = re.search(r"AssertionError:\s*(.+?)\s+is not\s+(.+)$", stripped)
        if match:
            return match.group(2), match.group(1)

        match = re.search(r"AssertionError:\s*(.+?)\s+is not true", stripped, re.I)
        if match:
            return "True", match.group(1)

        match = re.search(r"AssertionError:\s*(.+?)\s+is not false", stripped, re.I)
        if match:
            return "False", match.group(1)

    joined = "\n".join(lines)
    match = re.search(r"Lists differ:\s*(.*?)\s*!=\s*(.*)", joined)
    if match:
        return match.group(2), match.group(1)

    match = re.search(r"Tuples differ:\s*(.*?)\s*!=\s*(.*)", joined)
    if match:
        return match.group(2), match.group(1)

    return "", ""


def normalize_method_name(value: str) -> str:
    value = re.sub(r"^test_", "", text(value))
    value = re.sub(r"_\d+$", "", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower()


def task_compare(vanilla: JsonDict | None, contract_guided: JsonDict | None) -> str:
    v_pass = vanilla is not None and vanilla.get("passed") is True
    c_pass = contract_guided is not None and contract_guided.get("passed") is True

    if v_pass and c_pass:
        return "passed_both"
    if v_pass and not c_pass:
        return "passed_vanilla_failed_contract_guided"
    if not v_pass and c_pass:
        return "failed_vanilla_passed_contract_guided"
    return "failed_both"


def method_contracts(contract: JsonDict) -> dict[str, JsonDict]:
    result = {}
    for method in as_list(contract.get("method_contracts")):
        if isinstance(method, dict):
            name = text(method.get("method_name"))
            if name:
                result[name] = method
    return result


def method_code_excerpt(code: str, method_name: str, limit: int = 2500) -> str:
    if not code or not method_name:
        return ""

    pattern = re.compile(
        rf"(^[ \t]*(?:@\w+(?:\([^)]*\))?\s*\n[ \t]*)*def\s+{re.escape(method_name)}\s*\(.*?)(?=^[ \t]*(?:@\w|def\s+|class\s+)|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(code)
    return short(match.group(1), limit) if match else ""


def create_analysis_prompt(
    benchmark: str,
    task_id: str,
    dataset_task: JsonDict,
    comparison_status: str,
    vanilla_detail: JsonDict | None,
    vanilla_generation: JsonDict,
    contract_detail: JsonDict | None,
    contract_generation: JsonDict,
    raw_contract: JsonDict,
    include_code_limit: int,
) -> str:
    vanilla_eval = as_dict((vanilla_detail or {}).get("evaluation"))
    contract_eval = as_dict((contract_detail or {}).get("evaluation"))

    vanilla_code = text(vanilla_generation.get("generated_code") or vanilla_generation.get("code"))
    contract_code = text(contract_generation.get("generated_code") or contract_generation.get("code"))

    payload = {
        "benchmark": benchmark,
        "task_id": task_id,
        "comparison_status": comparison_status,
        "dataset_task": dataset_task,
        "vanilla": {
            "passed": (vanilla_detail or {}).get("passed"),
            "failure_type": (vanilla_detail or {}).get("failure_type"),
            "summary_error": short((vanilla_detail or {}).get("error"), 3000),
            "failed_tests": parsed_failure_blocks(
                text(vanilla_eval.get("stderr") or (vanilla_detail or {}).get("error"))
            ),
            "generated_code": short(vanilla_code, include_code_limit),
        },
        "contract_guided": {
            "passed": (contract_detail or {}).get("passed"),
            "failure_type": (contract_detail or {}).get("failure_type"),
            "summary_error": short((contract_detail or {}).get("error"), 3000),
            "failed_tests": parsed_failure_blocks(
                text(contract_eval.get("stderr") or (contract_detail or {}).get("error"))
            ),
            "raw_contract": raw_contract,
            "generated_code": short(contract_code, include_code_limit),
        },
    }

    return f"""
You are a senior Python benchmark failure analyst.

You are doing post-hoc analysis only.
Do not generate a replacement solution.
Do not repair code.
Do not use hidden assumptions.
Use only the dataset task, generated outputs, official evaluation results, failed test traces, and raw contract provided below.

Goal:
Analyze why the task failed in vanilla, contract-guided, or both.
Identify why a task passed in vanilla but failed in contract-guided, or failed in vanilla but passed in contract-guided.
For each failed test, identify:
- failed pipeline: vanilla or contract_guided
- failed test class and test name
- target method/function
- related contract type
- expected result
- actual result
- precise reason for failure
- whether the raw contract helped, hurt, was incomplete, was wrong, or was not relevant
- what conceptually should change in the generated implementation
- evidence from the traceback/output/code/contract

Allowed related_contract_type values:
{CONTRACT_TYPES}

Return exactly one valid JSON object with this schema:
{{
  "task_id": "...",
  "comparison_status": "...",
  "task_level_reason": "...",
  "vanilla_overall_reason": "...",
  "contract_guided_overall_reason": "...",
  "contract_effect": "helped | hurt | neutral | mixed | not_applicable",
  "rows": [
    {{
      "pipeline": "vanilla | contract_guided",
      "test_class": "...",
      "test_name": "...",
      "target_method": "...",
      "related_contract_type": "...",
      "contract_clause_or_gap": "...",
      "expected_result": "...",
      "actual_result": "...",
      "why_failed": "...",
      "solution_direction": "...",
      "evidence_from_results": "...",
      "evidence_from_code": "...",
      "evidence_from_contract": "..."
    }}
  ]
}}

Rules:
- If a pipeline passed, do not invent failed tests for it.
- If both passed, return rows as an empty array.
- If generation failed, create one row with pipeline and related_contract_type such as implementation completeness, syntax/formatting, or interface/signature.
- Be specific and concise.
- Do not mention that you are an AI model.
- Do not include markdown.
- Return JSON only.

ANALYSIS INPUT:
{dump_json(payload, 50000)}
""".strip()


def extract_json_object(text_value: str) -> JsonDict:
    content = text(text_value)
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(content[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def call_analyzer(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> JsonDict:
    response, _api_result = call_chat_model(
        client=client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python benchmark failure analyst. "
                    "Return exactly one valid JSON object. No markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )

    parsed = extract_json_object(response)
    if not parsed:
        raise ValueError(f"Analyzer did not return a valid JSON object: {short(response, 1000)}")
    return parsed


def cache_key(task_id: str, source_provider: str, source_model: str, analyzer_provider: str, analyzer_model: str) -> str:
    return "|".join([task_id, source_provider, source_model, analyzer_provider, analyzer_model])


def read_cache(path: Path) -> dict[str, JsonDict]:
    if not path.exists():
        return {}

    cache: dict[str, JsonDict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = text(row.get("_cache_key"))
        if key:
            cache[key] = row
    return cache


def append_cache(path: Path, key: str, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"_cache_key": key, **value}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_details(rows: list[JsonDict]) -> JsonDict:
    total = len(rows)
    passed = sum(row.get("passed") is True for row in rows)
    failures = Counter(text(row.get("failure_type")) or "passed" for row in rows)

    total_tests = 0
    passed_tests = 0
    for row in rows:
        metrics = as_dict(as_dict(row.get("evaluation")).get("metrics"))
        total_tests += metrics.get("total") or 0
        passed_tests += metrics.get("passed") or 0

    return {
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass@1": passed / total if total else 0.0,
        "pass@1_percent": round((passed / total * 100) if total else 0.0, 2),
        "generation_failed": failures.get("generation_failed", 0),
        "wrong_answer": failures.get("wrong_answer", 0),
        "type_error": failures.get("type_error", 0),
        "value_error": failures.get("value_error", 0),
        "key_error": failures.get("key_error", 0),
        "attribute_error": failures.get("attribute_error", 0),
        "timeout": failures.get("timeout", 0),
        "syntax_error": failures.get("syntax_error", 0),
        "executed_tests_total": total_tests,
        "executed_tests_passed": passed_tests,
        "executed_test_pass_rate": passed_tests / total_tests if total_tests else None,
    }


def build_summary_sheet(
    benchmark: str,
    source_provider: str,
    source_model: str,
    analyzer_provider: str,
    analyzer_model: str,
    vanilla_details: list[JsonDict],
    contract_details: list[JsonDict],
) -> pd.DataFrame:
    vanilla = summarize_details(vanilla_details)
    contract = summarize_details(contract_details)

    return pd.DataFrame(
        [
            {
                "benchmark": benchmark,
                "source_provider": source_provider,
                "source_model": source_model,
                "analyzer_provider": analyzer_provider,
                "analyzer_model": analyzer_model,
                "pipeline": "vanilla",
                **vanilla,
            },
            {
                "benchmark": benchmark,
                "source_provider": source_provider,
                "source_model": source_model,
                "analyzer_provider": analyzer_provider,
                "analyzer_model": analyzer_model,
                "pipeline": "contract_guided",
                **contract,
            },
            {
                "benchmark": benchmark,
                "source_provider": source_provider,
                "source_model": source_model,
                "analyzer_provider": analyzer_provider,
                "analyzer_model": analyzer_model,
                "pipeline": "contract_guided_minus_vanilla",
                "total_tasks": contract["total_tasks"],
                "passed": contract["passed"] - vanilla["passed"],
                "failed": contract["failed"] - vanilla["failed"],
                "pass@1": contract["pass@1"] - vanilla["pass@1"],
                "pass@1_percent": contract["pass@1_percent"] - vanilla["pass@1_percent"],
                "generation_failed": contract["generation_failed"] - vanilla["generation_failed"],
                "wrong_answer": contract["wrong_answer"] - vanilla["wrong_answer"],
                "type_error": contract["type_error"] - vanilla["type_error"],
                "value_error": contract["value_error"] - vanilla["value_error"],
                "key_error": contract["key_error"] - vanilla["key_error"],
                "attribute_error": contract["attribute_error"] - vanilla["attribute_error"],
                "timeout": contract["timeout"] - vanilla["timeout"],
                "syntax_error": contract["syntax_error"] - vanilla["syntax_error"],
                "executed_tests_total": contract["executed_tests_total"] - vanilla["executed_tests_total"],
                "executed_tests_passed": contract["executed_tests_passed"] - vanilla["executed_tests_passed"],
                "executed_test_pass_rate": None,
            },
        ]
    )


def contract_mapping_rows(task_id: str, contract: JsonDict) -> list[JsonDict]:
    rows = []
    for method_name, method in method_contracts(contract).items():
        body = as_dict(method.get("contract"))
        rows.append(
            {
                "task_id": task_id,
                "method_name": method_name,
                "signature": method.get("signature"),
                "dependencies": dump_json(method.get("dependencies"), 1200),
                "preconditions": dump_json(body.get("preconditions"), 2000),
                "postconditions": dump_json(body.get("postconditions"), 2000),
                "invariants": dump_json(body.get("invariants"), 1200),
                "edge_cases": dump_json(body.get("edge_cases"), 1200),
                "invalid_input_behavior": dump_json(body.get("invalid_input_behavior"), 1200),
                "output": dump_json(as_dict(body.get("interface")).get("output"), 1000),
            }
        )
    return rows


def write_excel(
    path: Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

            worksheet = writer.sheets[safe_sheet]
            worksheet.freeze_panes = "A2"

            for index, column in enumerate(df.columns, start=1):
                values = [len(str(column))]
                values.extend(min(len(str(value)), 80) for value in df[column].fillna("").head(500))
                width = min(max(max(values) + 2, 12), 70)
                worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a large LLM to perform post-hoc failure analysis over benchmark outputs. "
            "The analyzer does not generate or repair code."
        )
    )

    parser.add_argument("--root", default=".", help="Project root containing data/, outputs/, and results/.")
    parser.add_argument("--benchmark", default="classeval")

    parser.add_argument("--source-provider", required=True, help="Small LLM provider that generated the code.")
    parser.add_argument("--source-model", required=True, help="Small LLM model that generated the code.")

    parser.add_argument("--analyzer-provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--analyzer-model", help="Large LLM model used only for analysis.")

    parser.add_argument("--dataset-file")
    parser.add_argument("--vanilla-details")
    parser.add_argument("--contract-details")

    parser.add_argument("--include-test-source", action="store_true")
    parser.add_argument("--test-source-limit", type=int, default=6000)
    parser.add_argument("--code-limit", type=int, default=8000)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--max-tasks", type=int)

    parser.add_argument("--cache")
    parser.add_argument("--overwrite-cache", action="store_true")

    parser.add_argument("--out", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    benchmark = args.benchmark
    source_provider = args.source_provider
    source_model = args.source_model

    analyzer_model = args.analyzer_model or default_model(args.analyzer_provider)
    analyzer_client = get_client(args.analyzer_provider)

    dataset_by_task = load_dataset_by_task(root, benchmark, args.dataset_file)

    vanilla_details_path = find_details_file(
        root,
        benchmark,
        "vanilla",
        source_provider,
        source_model,
        args.vanilla_details,
    )
    contract_details_path = find_details_file(
        root,
        benchmark,
        "contract_guided",
        source_provider,
        source_model,
        args.contract_details,
    )

    vanilla_details = load_json(vanilla_details_path)
    contract_details = load_json(contract_details_path)

    if not isinstance(vanilla_details, list):
        raise ValueError(f"Vanilla details must be a JSON list: {vanilla_details_path}")
    if not isinstance(contract_details, list):
        raise ValueError(f"Contract-guided details must be a JSON list: {contract_details_path}")

    vanilla_by_task = {
        text(item.get("task_id")): item
        for item in vanilla_details
        if isinstance(item, dict)
    }
    contract_by_task = {
        text(item.get("task_id")): item
        for item in contract_details
        if isinstance(item, dict)
    }

    task_ids = sorted(set(vanilla_by_task) | set(contract_by_task))

    selected_task_ids = [
        task_id
        for task_id in task_ids
        if task_compare(vanilla_by_task.get(task_id), contract_by_task.get(task_id)) != "passed_both"
    ]

    if args.max_tasks:
        selected_task_ids = selected_task_ids[: args.max_tasks]

    cache_path = (
        Path(args.cache)
        if args.cache
        else root
        / "results"
        / benchmark
        / "analysis"
        / f"llm_analysis_cache_{safe_name(source_provider)}_{safe_name(source_model)}_by_{safe_name(args.analyzer_provider)}_{safe_name(analyzer_model)}.jsonl"
    )
    cache = {} if args.overwrite_cache else read_cache(cache_path)

    task_rows: list[JsonDict] = []
    analysis_rows: list[JsonDict] = []
    raw_failure_rows: list[JsonDict] = []
    contract_rows: list[JsonDict] = []

    for number, task_id in enumerate(selected_task_ids, start=1):
        vanilla_detail = vanilla_by_task.get(task_id)
        contract_detail = contract_by_task.get(task_id)

        comparison = task_compare(vanilla_detail, contract_detail)

        dataset_task = dataset_by_task.get(task_id, {})
        sanitized_dataset = remove_solution_fields(
            dataset_task,
            include_test_source=args.include_test_source,
            test_source_limit=args.test_source_limit,
        )

        vanilla_generation = load_generation_record(root, vanilla_detail or {})
        contract_generation = load_generation_record(root, contract_detail or {})
        raw_contract = load_contract(
            root,
            benchmark,
            source_provider,
            source_model,
            task_id,
            contract_generation,
        )

        task_rows.append(
            {
                "task_id": task_id,
                "class_or_entry_point": text(
                    (contract_detail or vanilla_detail or {}).get("class_name")
                    or (contract_detail or vanilla_detail or {}).get("entry_point")
                ),
                "comparison_status": comparison,
                "vanilla_passed": (vanilla_detail or {}).get("passed"),
                "contract_guided_passed": (contract_detail or {}).get("passed"),
                "vanilla_failure_type": (vanilla_detail or {}).get("failure_type"),
                "contract_guided_failure_type": (contract_detail or {}).get("failure_type"),
                "vanilla_error": short((vanilla_detail or {}).get("error"), 1200),
                "contract_guided_error": short((contract_detail or {}).get("error"), 1200),
            }
        )

        for pipeline, detail in (("vanilla", vanilla_detail), ("contract_guided", contract_detail)):
            if not detail or detail.get("passed") is True:
                continue

            evaluation = as_dict(detail.get("evaluation"))
            blocks = parsed_failure_blocks(text(evaluation.get("stderr") or detail.get("error")))
            if not blocks:
                blocks = [
                    {
                        "status": "GENERATION_FAILED" if detail.get("failure_type") == "generation_failed" else "FAILED",
                        "test_class": "",
                        "test_name": "",
                        "traceback": "",
                        "assertion_or_error": short(detail.get("error"), 1500),
                        "expected_result": "",
                        "actual_result": "",
                    }
                ]

            for block in blocks:
                raw_failure_rows.append(
                    {
                        "task_id": task_id,
                        "pipeline": pipeline,
                        "comparison_status": comparison,
                        "failure_type": detail.get("failure_type"),
                        **block,
                    }
                )

        contract_rows.extend(contract_mapping_rows(task_id, raw_contract))

        key = cache_key(task_id, source_provider, source_model, args.analyzer_provider, analyzer_model)

        if key in cache:
            analysis = cache[key]
        else:
            prompt = create_analysis_prompt(
                benchmark=benchmark,
                task_id=task_id,
                dataset_task=sanitized_dataset,
                comparison_status=comparison,
                vanilla_detail=vanilla_detail,
                vanilla_generation=vanilla_generation,
                contract_detail=contract_detail,
                contract_generation=contract_generation,
                raw_contract=raw_contract,
                include_code_limit=args.code_limit,
            )

            print(f"[{number}/{len(selected_task_ids)}] Analyzing {task_id} with {args.analyzer_provider}/{analyzer_model}")

            try:
                analysis = call_analyzer(
                    analyzer_client,
                    analyzer_model,
                    prompt,
                    args.temperature,
                    args.max_tokens,
                )
            except Exception as exc:
                analysis = {
                    "task_id": task_id,
                    "comparison_status": comparison,
                    "task_level_reason": f"Analyzer failed: {exc}",
                    "vanilla_overall_reason": "",
                    "contract_guided_overall_reason": "",
                    "contract_effect": "unknown",
                    "rows": [],
                }

            append_cache(cache_path, key, analysis)

            if args.delay:
                time.sleep(args.delay)

        rows = as_list(analysis.get("rows"))
        if not rows:
            analysis_rows.append(
                {
                    "task_id": task_id,
                    "comparison_status": comparison,
                    "pipeline": "",
                    "test_class": "",
                    "test_name": "",
                    "target_method": "",
                    "related_contract_type": "",
                    "contract_clause_or_gap": "",
                    "expected_result": "",
                    "actual_result": "",
                    "why_failed": analysis.get("task_level_reason", ""),
                    "solution_direction": "",
                    "contract_effect": analysis.get("contract_effect", ""),
                    "vanilla_overall_reason": analysis.get("vanilla_overall_reason", ""),
                    "contract_guided_overall_reason": analysis.get("contract_guided_overall_reason", ""),
                    "evidence_from_results": "",
                    "evidence_from_code": "",
                    "evidence_from_contract": "",
                }
            )
        else:
            for row in rows:
                if not isinstance(row, dict):
                    continue

                analysis_rows.append(
                    {
                        "task_id": task_id,
                        "comparison_status": comparison,
                        "pipeline": row.get("pipeline", ""),
                        "test_class": row.get("test_class", ""),
                        "test_name": row.get("test_name", ""),
                        "target_method": row.get("target_method", ""),
                        "related_contract_type": row.get("related_contract_type", ""),
                        "contract_clause_or_gap": row.get("contract_clause_or_gap", ""),
                        "expected_result": row.get("expected_result", ""),
                        "actual_result": row.get("actual_result", ""),
                        "why_failed": row.get("why_failed", ""),
                        "solution_direction": row.get("solution_direction", ""),
                        "contract_effect": analysis.get("contract_effect", ""),
                        "vanilla_overall_reason": analysis.get("vanilla_overall_reason", ""),
                        "contract_guided_overall_reason": analysis.get("contract_guided_overall_reason", ""),
                        "evidence_from_results": row.get("evidence_from_results", ""),
                        "evidence_from_code": row.get("evidence_from_code", ""),
                        "evidence_from_contract": row.get("evidence_from_contract", ""),
                    }
                )

    summary_df = build_summary_sheet(
        benchmark,
        source_provider,
        source_model,
        args.analyzer_provider,
        analyzer_model,
        vanilla_details,
        contract_details,
    )

    sheets = {
        "Summary": summary_df,
        "Task Comparison": pd.DataFrame(task_rows),
        "LLM Failure Analysis": pd.DataFrame(analysis_rows),
        "Raw Failed Tests": pd.DataFrame(raw_failure_rows),
        "Contract Mapping": pd.DataFrame(contract_rows),
    }

    write_excel(Path(args.out), sheets)

    print(f"Vanilla details:         {vanilla_details_path}")
    print(f"Contract-guided details: {contract_details_path}")
    print(f"Dataset tasks:           {len(dataset_by_task)}")
    print(f"Analyzed tasks:          {len(selected_task_ids)}")
    print(f"Cache:                   {cache_path}")
    print(f"Excel report:            {args.out}")


if __name__ == "__main__":
    main()