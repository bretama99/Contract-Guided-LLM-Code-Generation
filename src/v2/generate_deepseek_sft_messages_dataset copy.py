from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = "Generate compact Python contracts. Output JSON only."

OUTPUT_FIELDS = (
    "preconditions",
    "postconditions",
    "invariants",
    "edge_cases",
    "invalid_input_behavior",
)

PROMPT_FIELDS = ("complete_prompt", "instruct_prompt", "prompt")

EXCLUDED_FIELDS = {
    "test",
    "tests",
    "canonical_solution",
    "solution",
    "reference_solution",
    "code_prompt",
    "generated_code",
    "completion",
    "evaluation",
    "eval_result",
    "failure_reason",
    "repair_feedback",
}

LEAKAGE_FIELDS = EXCLUDED_FIELDS - {"code_prompt"}

SOURCE_VALUES = {
    "signature",
    "type_hint",
    "explicit",
    "example",
    "strongly_implied",
    "inferred",
    "not_specified",
}

FORBIDDEN_TARGET_TEXT = (
    "```",
    "schema_version",
    "schema_type",
    "task_id",
    "benchmark",
    "metadata",
    "canonical_solution",
    "reference_solution",
    "hidden test",
    "hidden_test",
    "unittest",
    "TestCases",
    "self.assert",
    "assertEqual",
    "assertRaises",
    "pytest",
    "def task_func",
    "class Test",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("tasks", "records", "data", "items"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]

    raise ValueError(f"Expected JSON list of task objects: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

def pick_prompt(task: dict[str, Any]) -> str:
    for key in PROMPT_FIELDS:
        value = s(task.get(key))
        if value:
            return value
    return ""


def parse_libs(task: dict[str, Any]) -> list[str]:
    libs = task.get("libs")

    if isinstance(libs, list):
        return [s(x) for x in libs if s(x)]

    if isinstance(libs, str):
        raw = libs.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [s(x) for x in parsed if s(x)]
        except Exception:
            pass
        return [raw]

    return []


def make_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": pick_prompt(task),
        "libs": parse_libs(task),
    }


def build_task_context(task: dict[str, Any]) -> str:
    prompt = s(task["prompt"])
    libs = [lib for lib in task.get("libs", []) if lib and lib not in prompt]

    text = "Task:\n" + prompt
    if libs:
        text += "\n\nLibraries: " + ", ".join(libs)
    return text.strip()


def build_user_prompt(task: dict[str, Any]) -> str:
    return (
        build_task_context(task)
        + "\n\nReturn only compact JSON with this top-level structure:\n"
        + '{"preconditions":[],"postconditions":[],"invariants":[],"edge_cases":[],'
        + '"invalid_input_behavior":{"specified":false,"behavior":"not specified",'
        + '"exception_type":"","description":"","source":"not_specified"}}'
        + "\n\npostconditions must describe the expected output."
        + "\nUse [] for preconditions, invariants, or edge_cases when unsupported by the task."
    )


def build_teacher_prompt(task: dict[str, Any]) -> str:
    return f"""Generate a compact behavioral contract for one Python function.

Use only the visible task description, signature, docstring, imports, examples, and stated constraints.
Do not use tests, canonical solutions, generated code, execution results, hidden-test behavior, or benchmark internals.

Return JSON only with exactly these top-level keys:
preconditions, postconditions, invariants, edge_cases, invalid_input_behavior.

Required JSON shape:
{{
  "preconditions": [
    {{"id":"P1","target":"input name or call","description":"valid input assumption or caller obligation","source":"explicit"}}
  ],
  "postconditions": [
    {{"id":"Q1","target":"return","description":"observable output guarantee after successful execution","source":"explicit"}}
  ],
  "invariants": [
    {{"id":"I1","description":"stable input/output property, only if meaningful for this task","source":"strongly_implied"}}
  ],
  "edge_cases": [
    {{"id":"E1","case":"valid special input or situation","expected_behavior":"expected behavior","source":"explicit"}}
  ],
  "invalid_input_behavior": {{
    "specified": false,
    "behavior": "not specified",
    "exception_type": "",
    "description": "The task does not specify invalid-input behavior.",
    "source": "not_specified"
  }}
}}

Rules:
- postconditions must be non-empty.
- preconditions may be [] if no valid input assumption is stated or strongly implied.
- invariants may be [] if the task has no meaningful invariant.
- edge_cases may be [] if no valid special case is stated or strongly implied.
- invalid_input_behavior.specified must be true only when invalid-input behavior is explicitly stated.
- Do not output code, tests, metadata, schema_version, schema_type, task_id, or benchmark.
- Do not invent exceptions, hidden requirements, unsupported constraints, or implementation details.
- Keep each clause short, precise, faithful, and useful for code generation.

{build_task_context(task)}""".strip()


def extract_json(raw: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found")

    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(raw[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(raw[start : i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("Extracted JSON is not an object")
                return obj

    raise ValueError("Incomplete JSON object")


def first(*values: Any) -> str:
    for value in values:
        value = s(value)
        if value:
            return value
    return ""


def norm_source(value: Any, default: str = "inferred") -> str:
    value = s(value).lower()
    return value if value in SOURCE_VALUES else default


def norm_list(value: Any, prefix: str, kind: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    out = []
    for i, item in enumerate(value, 1):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, dict):
            continue

        row: dict[str, Any] = {"id": s(item.get("id")) or f"{prefix}{i}"}

        if kind == "edge":
            row["case"] = first(item.get("case"), item.get("input"), item.get("description"))
            row["expected_behavior"] = first(
                item.get("expected_behavior"),
                item.get("behavior"),
                item.get("description"),
            )
        elif kind == "invariant":
            row["description"] = first(
                item.get("description"),
                item.get("property"),
                item.get("invariant"),
            )
        else:
            row["target"] = first(
                item.get("target"),
                item.get("input"),
                item.get("name"),
                "return" if prefix == "Q" else "input",
            )
            row["description"] = first(
                item.get("description"),
                item.get("condition"),
                item.get("guarantee"),
                item.get("behavior"),
            )

        row["source"] = norm_source(item.get("source"))
        out.append(row)

    return out


def norm_invalid(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "specified": False,
            "behavior": "not specified",
            "exception_type": "",
            "description": "The task does not specify invalid-input behavior.",
            "source": "not_specified",
        }

    specified = value.get("specified")
    specified = specified.strip().lower() in {"true", "yes", "1"} if isinstance(specified, str) else bool(specified)

    if not specified:
        return {
            "specified": False,
            "behavior": "not specified",
            "exception_type": "",
            "description": s(value.get("description")) or "The task does not specify invalid-input behavior.",
            "source": "not_specified",
        }

    return {
        "specified": True,
        "behavior": first(value.get("behavior"), value.get("expected_behavior")),
        "exception_type": s(value.get("exception_type")),
        "description": s(value.get("description")),
        "source": norm_source(value.get("source"), "explicit"),
    }


def normalize_contract(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("contract"), dict):
        raw = raw["contract"]

    return {
        "preconditions": norm_list(raw.get("preconditions"), "P", "pre"),
        "postconditions": norm_list(raw.get("postconditions"), "Q", "post"),
        "invariants": norm_list(raw.get("invariants"), "I", "invariant"),
        "edge_cases": norm_list(raw.get("edge_cases"), "E", "edge"),
        "invalid_input_behavior": norm_invalid(raw.get("invalid_input_behavior")),
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if tuple(contract.keys()) != OUTPUT_FIELDS:
        raise ValueError(f"Contract must contain exactly: {', '.join(OUTPUT_FIELDS)}")

    for field in ("preconditions", "postconditions", "invariants", "edge_cases"):
        if not isinstance(contract[field], list):
            raise ValueError(f"{field} must be a list")

    if not contract["postconditions"]:
        raise ValueError("postconditions must be non-empty")

    for item in contract["preconditions"]:
        if not isinstance(item, dict) or not first(item.get("target"), item.get("description")):
            raise ValueError("Invalid precondition object")
        if not s(item.get("target")) or not s(item.get("description")):
            raise ValueError("Each precondition needs target and description")

    for item in contract["postconditions"]:
        if not isinstance(item, dict) or not s(item.get("target")) or not s(item.get("description")):
            raise ValueError("Each postcondition needs target and description")

    for item in contract["invariants"]:
        if not isinstance(item, dict) or not s(item.get("description")):
            raise ValueError("Each invariant needs description")

    for item in contract["edge_cases"]:
        if not isinstance(item, dict) or not s(item.get("case")) or not s(item.get("expected_behavior")):
            raise ValueError("Each edge case needs case and expected_behavior")

    invalid = contract["invalid_input_behavior"]
    if not isinstance(invalid, dict):
        raise ValueError("invalid_input_behavior must be an object")
    for key in ("specified", "behavior", "exception_type", "description", "source"):
        if key not in invalid:
            raise ValueError(f"invalid_input_behavior missing key: {key}")
    if not isinstance(invalid["specified"], bool):
        raise ValueError("invalid_input_behavior.specified must be boolean")

    target = json.dumps(contract, ensure_ascii=False).lower()
    for bad in FORBIDDEN_TARGET_TEXT:
        if bad.lower() in target:
            raise ValueError(f"Forbidden target text found: {bad}")


def contains_snippet(haystack: str, needle: Any, n: int = 120) -> bool:
    needle = s(needle)
    return len(needle) >= n and (needle[:n] in haystack or needle[-n:] in haystack)


def check_no_leakage(original: dict[str, Any], messages: list[dict[str, str]]) -> None:
    combined = "\n".join(m["content"] for m in messages)
    for field in LEAKAGE_FIELDS:
        if field in original and contains_snippet(combined, original[field]):
            raise ValueError(f"Excluded field leaked into dataset: {field}")


def call_teacher(client: Any, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    from src.common.llm_clients import call_chat_model

    raw, _info = call_chat_model(
        client=client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    return raw


def generate_contract(
    client: Any,
    model: str,
    task: dict[str, Any],
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> tuple[dict[str, Any], int]:
    prompt = build_teacher_prompt(task)
    current = prompt
    last_error = ""
    last_raw = ""

    for attempt in range(1, max_retries + 2):
        last_raw = call_teacher(client, model, current, temperature, max_tokens)

        try:
            contract = normalize_contract(extract_json(last_raw))
            validate_contract(contract)
            return contract, attempt
        except Exception as exc:
            last_error = str(exc)
            current = (
                prompt
                + "\n\nYour previous output failed validation: "
                + last_error
                + "\nReturn only valid JSON. postconditions must be non-empty. "
                + "preconditions, invariants, and edge_cases may be [] only when unsupported."
            )

    raise ValueError(
        f"Teacher failed after {max_retries + 1} attempts: {last_error}. "
        f"Last raw response: {last_raw[:1000]!r}"
    )


def make_messages(task: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
        {"role": "assistant", "content": json.dumps(contract, ensure_ascii=False, separators=(",", ":"))},
    ]


def split_rows(rows: list[dict[str, Any]], seed: int, val_ratio: float, val_count: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)

    if not rows:
        return [], []

    if val_count is None:
        val_count = int(round(len(rows) * val_ratio))
        if val_ratio > 0 and len(rows) > 1:
            val_count = max(1, val_count)

    val_count = max(0, min(val_count, len(rows) - 1 if len(rows) > 1 else 0))
    return rows[val_count:], rows[:val_count]


def messages_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": row["messages"]} for row in rows]


def preview(tasks: list[dict[str, Any]]) -> None:
    if not tasks:
        return

    task = make_task(tasks[0])
    sample_contract = {
        "preconditions": [],
        "postconditions": [
            {
                "id": "Q1",
                "target": "return",
                "description": "The returned value satisfies the task output requirement.",
                "source": "explicit",
            }
        ],
        "invariants": [],
        "edge_cases": [],
        "invalid_input_behavior": {
            "specified": False,
            "behavior": "not specified",
            "exception_type": "",
            "description": "The task does not specify invalid-input behavior.",
            "source": "not_specified",
        },
    }

    print("\n--- train/val row shape ---")
    print(json.dumps({"messages": make_messages(task, sample_contract)[:2]}, indent=2, ensure_ascii=False))

    print("\n--- student user prompt preview ---")
    print(build_user_prompt(task)[:4000])

    print("\n--- teacher prompt preview ---")
    print(build_teacher_prompt(task)[:4000])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DeepSeek compact contract SFT dataset.")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1800)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-count", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.val_count is not None and args.val_count < 0:
        raise ValueError("--val-count must be non-negative")

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    all_path = out_dir / "all.jsonl"
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    errors_path = out_dir / "errors.jsonl"
    summary_path = out_dir / "summary.json"

    tasks = load_tasks(input_path)
    selected = tasks[args.start : args.start + args.limit if args.limit is not None else None]

    print(f"[INFO] Loaded records: {len(tasks)}")
    print(f"[INFO] Selected records: {len(selected)}")
    print(f"[INFO] Output dir: {out_dir}")

    if args.dry_run:
        preview(selected)
        return

    output_files = (all_path, train_path, val_path, errors_path, summary_path)
    if out_dir.exists() and any(p.exists() for p in output_files) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already has output files. Use --overwrite.")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_files:
            if path.exists():
                path.unlink()

    from src.common.llm_clients import default_model, get_client

    model = args.model or default_model(args.provider)
    client = get_client(args.provider)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for index, original in enumerate(selected, start=args.start):
        task = make_task(original)
        print(f"[{index}] GENERATE ", flush=True)

        try:
            if not task["prompt"]:
                raise ValueError("Empty task prompt")

            contract, attempts = generate_contract(
                client=client,
                model=model,
                task=task,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )

            messages = make_messages(task, contract)
            check_no_leakage(original, messages)

            row = {"id": len(rows) + 1, "messages": messages}
            rows.append(row)
            append_jsonl(all_path, row)

        except Exception as exc:
            error = {
                "id": len(errors) + 1,
                "error": str(exc),
                "created_at": now(),
            }
            errors.append(error)
            append_jsonl(errors_path, error)

        if args.delay > 0:
            time.sleep(args.delay)

    train_rows, val_rows = split_rows(rows, args.seed, args.val_ratio, args.val_count)
    write_jsonl(train_path, messages_only(train_rows))
    write_jsonl(val_path, messages_only(val_rows))

    summary = {
        "created_at": now(),
        "input": str(input_path),
        "output_dir": str(out_dir),
        "relationship": "task_description_to_compact_contract_json",
        "format": "train_val_messages_only",
        "teacher_provider": args.provider,
        "teacher_model": model,
        "selected_records": len(selected),
        "valid_records": len(rows),
        "error_records": len(errors),
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "excluded_fields": sorted(EXCLUDED_FIELDS),
        "train_val_fields": ["messages"],
        "all_jsonl_fields": ["id", "messages"],
        "assistant_contract_fields": list(OUTPUT_FIELDS),
        "quality_checks": {
            "postconditions_non_empty": True,
            "preconditions_may_be_empty": True,
            "invariants_may_be_empty": True,
            "edge_cases_may_be_empty": True,
            "invalid_input_behavior_object_required": True,
            "no_metadata_in_train_val": True,
            "no_schema_version": True,
            "no_schema_type": True,
            "tests_and_canonical_solutions_excluded": True,
        },
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
    }

    write_json(summary_path, summary)

    print("\n[DONE]")
    print(f"Valid: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Train: {len(train_rows)}")
    print(f"Val: {len(val_rows)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()