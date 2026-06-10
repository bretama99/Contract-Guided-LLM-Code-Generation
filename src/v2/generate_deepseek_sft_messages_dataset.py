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

from src.common.llm_clients import call_chat_model, default_model, get_client

FIELDS = ("preconditions", "postconditions", "invariants", "edge_cases", "invalid_input_behavior")
PROMPT_FIELDS = ("complete_prompt", "instruct_prompt", "prompt")
EXCLUDED = {
    "test", "tests", "canonical_solution", "solution", "reference_solution",
    "code_prompt", "generated_code", "completion", "evaluation", "eval_result",
    "failure_reason", "repair_feedback",
}
LEAKAGE_FIELDS = EXCLUDED - {"code_prompt"}
SOURCES = {"signature", "type_hint", "explicit", "example", "strongly_implied", "inferred", "not_specified"}
FORBIDDEN = (
    "```", "schema_version", "schema_type", "metadata", "canonical_solution",
    "reference_solution", "hidden test", "hidden_test", "unittest", "testcases",
    "self.assert", "assertequal", "assertraises", "pytest",
)

SCHEMA = (
    '{"preconditions":[{"id":"P1","target":"input name or call",'
    '"description":"valid input assumption or caller obligation","source":"explicit"}],'
    '"postconditions":[{"id":"Q1","target":"return",'
    '"description":"concrete observable output guarantee","source":"explicit"}],'
    '"invariants":[{"id":"I1","description":"meaningful stable input/output property",'
    '"source":"strongly_implied"}],'
    '"edge_cases":[{"id":"E1","case":"valid special input or situation",'
    '"expected_behavior":"expected behavior","source":"explicit"}],'
    '"invalid_input_behavior":{"specified":false,"behavior":"not specified",'
    '"exception_type":"","description":"The task does not specify invalid-input behavior.",'
    '"source":"not_specified"}}'
)

SYSTEM_PROMPT = (
    "Generate compact Python behavioral contracts. Output JSON only. "
    "Use exactly this schema: " + SCHEMA + " "
    "preconditions: valid caller/input assumptions. "
    "postconditions: concrete checkable output guarantees. "
    "invariants: meaningful stable input/output properties only. "
    "edge_cases: valid special cases only. "
    "invalid_input_behavior: explicit invalid-input behavior only. "
    "postconditions must be non-empty; other lists may be [] when unsupported."
)
TEACHER_SYSTEM = "You generate behavioral contracts for Python functions. Output only valid JSON."


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def t(x: Any) -> str:
    return "" if x is None else str(x).strip()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("tasks", "records", "data", "items"):
            if isinstance(data.get(k), list):
                return [x for x in data[k] if isinstance(x, dict)]
    raise ValueError(f"Expected JSON list of task objects: {path}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

def task_id(task: dict[str, Any]) -> int | str:
    raw = t(task.get("task_id")) or t(task.get("id"))
    if raw and "/" in raw:
        suffix = raw.rsplit("/", 1)[-1]
        if suffix.isdigit():
            return int(suffix) + 1
    if raw.isdigit():
        return int(raw) + 1
    return raw or "unknown"

def prompt_of(task: dict[str, Any]) -> str:
    for k in PROMPT_FIELDS:
        v = t(task.get(k))
        if v:
            return v
    return ""


def libs_of(task: dict[str, Any]) -> list[str]:
    raw = task.get("libs")
    if isinstance(raw, list):
        return [t(x) for x in raw if t(x)]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [t(x) for x in parsed if t(x)]
        except Exception:
            pass
        return [raw.strip()]
    return []


def raises_of(task: dict[str, Any]) -> list[str]:
    raw = task.get("doc_struct")
    if not raw:
        return []
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
        vals = obj.get("raises", []) if isinstance(obj, dict) else []
        return [t(x) for x in vals if t(x)]
    except Exception:
        return []


def compact_task(task: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task_id(task), "prompt": prompt_of(task), "libs": libs_of(task), "raises": raises_of(task)}


def task_context(task: dict[str, Any]) -> str:
    prompt = t(task["prompt"])
    libs = [x for x in task.get("libs", []) if x and x not in prompt]
    raises = [x for x in task.get("raises", []) if x and x not in prompt]
    parts = ["Task:", prompt]
    if libs:
        parts += ["", "Libraries: " + ", ".join(libs)]
    if raises:
        parts += ["", "Explicit raises: " + "; ".join(raises)]
    return "\n".join(parts).strip()


def build_user_prompt(task: dict[str, Any]) -> str:
    return (
        task_context(task)
        + "\n\nReturn only compact JSON using exactly this schema:\n"
        + SCHEMA
        + "\n\nRules: postconditions must describe the expected output. "
        + "Use [] for preconditions, invariants, or edge_cases when unsupported by the task."
    )

def build_teacher_prompt(task: dict[str, Any]) -> str:
    return (
        "Generate a compact behavioral contract for one Python function.\n\n"
        "Use only the visible task description, signature, docstring, imports, examples, and stated constraints.\n"
        "Do not use tests, canonical solutions, generated code, execution results, hidden-test behavior, or benchmark internals.\n\n"
        "Return JSON only using exactly this schema:\n" + SCHEMA + "\n\n"
        "Quality rules:\n"
        "- Do not merely restate the problem description.\n"
        "- Prefer short atomic clauses: one behavior per clause.\n"
        "- Postconditions must be non-empty, concrete, checkable, and useful for code generation.\n"
        "- Mention measurable output relationships when supported: length, keys, values, counts, ordering, ranges, files, mutations, or exceptions.\n"
        "- Do not put output guarantees inside invariants; put them in postconditions.\n"
        "- Use invariants only for meaningful stable properties. Otherwise use [].\n"
        "- Do not copy ordinary examples into edge_cases unless the example reveals boundary or special-case behavior.\n"
        "- Edge cases must describe valid special cases, not ordinary examples and not invalid inputs.\n"
        "- Preconditions may be [] if no valid caller/input assumption is stated or strongly implied.\n"
        "- Avoid vague clauses such as 'valid input', 'correct result', 'as described', or 'satisfies the task'.\n"
        "- invalid_input_behavior.specified must be true only when invalid-input behavior is explicitly stated.\n"
        "- Do not invent exceptions, hidden requirements, unsupported constraints, or implementation details.\n"
        "- Do not output code, tests, metadata, schema_version, schema_type, task_id, or benchmark.\n\n"
        + task_context(task)
    )
    
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
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(raw[start:], start):
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
                obj = json.loads(raw[start:i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("Extracted JSON is not an object")
                return obj
    raise ValueError("Incomplete JSON object")


def first(*xs: Any) -> str:
    for x in xs:
        x = t(x)
        if x:
            return x
    return ""


def src(x: Any, default: str = "inferred") -> str:
    x = t(x).lower()
    return x if x in SOURCES else default


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
        row: dict[str, Any] = {"id": t(item.get("id")) or f"{prefix}{i}"}
        if kind == "edge":
            row["case"] = first(item.get("case"), item.get("input"), item.get("description"))
            row["expected_behavior"] = first(item.get("expected_behavior"), item.get("behavior"), item.get("description"))
        elif kind == "inv":
            row["description"] = first(item.get("description"), item.get("property"), item.get("invariant"))
        else:
            row["target"] = first(item.get("target"), item.get("input"), item.get("name"), "return" if prefix == "Q" else "input")
            row["description"] = first(item.get("description"), item.get("condition"), item.get("guarantee"), item.get("behavior"))
        row["source"] = src(item.get("source"))
        out.append(row)
    return out


def norm_invalid(value: Any) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    specified = value.get("specified")
    specified = specified.strip().lower() in {"true", "yes", "1"} if isinstance(specified, str) else bool(specified)
    if not specified:
        return {
            "specified": False,
            "behavior": "not specified",
            "exception_type": "",
            "description": first(value.get("description"), "The task does not specify invalid-input behavior."),
            "source": "not_specified",
        }
    return {
        "specified": True,
        "behavior": first(value.get("behavior"), value.get("expected_behavior")),
        "exception_type": t(value.get("exception_type")),
        "description": t(value.get("description")),
        "source": src(value.get("source"), "explicit"),
    }


def normalize_contract(raw: dict[str, Any]) -> dict[str, Any]:
    raw = raw.get("contract", raw) if isinstance(raw.get("contract"), dict) else raw
    return {
        "preconditions": norm_list(raw.get("preconditions"), "P", "pre"),
        "postconditions": norm_list(raw.get("postconditions"), "Q", "post"),
        "invariants": norm_list(raw.get("invariants"), "I", "inv"),
        "edge_cases": norm_list(raw.get("edge_cases"), "E", "edge"),
        "invalid_input_behavior": norm_invalid(raw.get("invalid_input_behavior")),
    }


def validate_contract(c: dict[str, Any]) -> None:
    if set(c) != set(FIELDS):
        raise ValueError(f"Contract must contain exactly: {', '.join(FIELDS)}")
    for f in ("preconditions", "postconditions", "invariants", "edge_cases"):
        if not isinstance(c[f], list):
            raise ValueError(f"{f} must be a list")
    if not c["postconditions"]:
        raise ValueError("postconditions must be non-empty")
    for x in c["preconditions"]:
        if not isinstance(x, dict) or not t(x.get("target")) or not t(x.get("description")):
            raise ValueError("Each precondition needs target and description")
    for x in c["postconditions"]:
        if not isinstance(x, dict) or not t(x.get("target")) or not t(x.get("description")):
            raise ValueError("Each postcondition needs target and description")
    for x in c["invariants"]:
        if not isinstance(x, dict) or not t(x.get("description")):
            raise ValueError("Each invariant needs description")
    for x in c["edge_cases"]:
        if not isinstance(x, dict) or not t(x.get("case")) or not t(x.get("expected_behavior")):
            raise ValueError("Each edge case needs case and expected_behavior")
    invalid = c["invalid_input_behavior"]
    if not isinstance(invalid, dict) or not isinstance(invalid.get("specified"), bool):
        raise ValueError("invalid_input_behavior must be an object with boolean specified")
    for k in ("behavior", "exception_type", "description", "source"):
        if k not in invalid:
            raise ValueError(f"invalid_input_behavior missing key: {k}")
    target = json.dumps(c, ensure_ascii=False).lower()
    for bad in FORBIDDEN:
        if bad in target:
            raise ValueError(f"Forbidden target text found: {bad}")


def has_snippet(haystack: str, needle: Any, n: int = 120) -> bool:
    needle = t(needle)
    return len(needle) >= n and (needle[:n] in haystack or needle[-n:] in haystack)


def check_no_leakage(original: dict[str, Any], messages: list[dict[str, str]]) -> None:
    combined = "\n".join(m["content"] for m in messages)
    for f in LEAKAGE_FIELDS:
        if f in original and has_snippet(combined, original[f]):
            raise ValueError(f"Excluded field leaked into dataset: {f}")


def ask_teacher(client: Any, model: str, prompt: str, args: argparse.Namespace) -> str:
    raw, _ = call_chat_model(
        client=client,
        model=model,
        messages=[{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return raw


def generate_contract(client: Any, model: str, task: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    base, prompt, last_error, last_raw = build_teacher_prompt(task), "", "", ""
    prompt = base
    for attempt in range(1, args.max_retries + 2):
        last_raw = ask_teacher(client, model, prompt, args)
        try:
            c = normalize_contract(extract_json(last_raw))
            validate_contract(c)
            return c, attempt
        except Exception as exc:
            last_error = str(exc)
            prompt = (
                base + "\n\nPrevious output failed validation: " + last_error
                + "\nReturn only valid JSON. Make postconditions concrete and non-empty. "
                + "Do not merely restate the task."
            )
    raise ValueError(f"Teacher failed after {args.max_retries + 1} attempts: {last_error}. Raw: {last_raw[:1000]!r}")


def make_messages(task: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
        {"role": "assistant", "content": json.dumps(contract, ensure_ascii=False, separators=(",", ":"))},
    ]


def split_map(tasks: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    path = out_dir / "split.json"
    if args.resume:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        ids = [task_id(x) for x in tasks]
        random.Random(args.seed).shuffle(ids)
        n = int(round(len(ids) * args.val_ratio))
        n = max(1, n) if args.val_ratio > 0 and len(ids) > 1 else n
        data = {"val": ids[:n], "train": ids[n:]}
        write_json(path, data)
    return {tid: split for split, ids in data.items() for tid in ids}


def written_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def materialize(all_path: Path, split: dict[str, str], train_path: Path, val_path: Path) -> None:
    rows = {"train": [], "val": []}
    if all_path.exists():
        with all_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows[split[row["task_id"]]].append({"messages": row["messages"]})
    write_jsonl(train_path, rows["train"])
    write_jsonl(val_path, rows["val"])


def preview(tasks: list[dict[str, Any]]) -> None:
    task = compact_task(tasks[0])
    sample = {
        "preconditions": [],
        "postconditions": [{"id": "Q1", "target": "return", "description": "The returned value satisfies the task output requirement.", "source": "explicit"}],
        "invariants": [],
        "edge_cases": [],
        "invalid_input_behavior": {"specified": False, "behavior": "not specified", "exception_type": "", "description": "The task does not specify invalid-input behavior.", "source": "not_specified"},
    }
    print("\n--- train/val row shape ---")
    print(json.dumps({"messages": make_messages(task, sample)}, indent=2, ensure_ascii=False)[:4000])
    print("\n--- teacher prompt preview ---")
    print(build_teacher_prompt(task)[:4000])

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--model")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1800)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0,1)")
    in_path, out_dir = Path(args.input), Path(args.output_dir)
    all_path, train_path, val_path, err_path = out_dir / "all.jsonl", out_dir / "train.jsonl", out_dir / "val.jsonl", out_dir / "errors.jsonl"
    all_tasks = load_tasks(in_path)
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    tasks = all_tasks[: args.limit] if args.limit is not None else all_tasks

    print(f"[INFO] Loaded records: {len(all_tasks)}")
    print(f"[INFO] Selected records: {len(tasks)}")
    print(f"[INFO] Output dir: {out_dir}")
    if args.dry_run:
        preview(tasks)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(p.exists() for p in (all_path, train_path, val_path, err_path, out_dir / "split.json")):
        raise FileExistsError(f"{out_dir} already has outputs. Use --resume or a new output dir.")
    split = split_map(tasks, out_dir, args)
    done = written_ids(all_path)
    client, model = get_client(args.provider), args.model or default_model(args.provider)
    errors, started = 0, time.perf_counter()
    for i, original in enumerate(tasks):
        tid = task_id(original)
        if tid in done:
            continue
        task = compact_task(original)
        print(f"[{i}] {tid} GENERATE", flush=True)
        try:
            if not task["prompt"]:
                raise ValueError("empty complete_prompt/instruct_prompt/prompt")
            contract, attempts = generate_contract(client, model, task, args)
            messages = make_messages(task, contract)
            check_no_leakage(original, messages)
            append_jsonl(all_path, {"task_id": tid, "messages": messages})
            done.add(tid)
            print(f"[{i}] {tid} DONE attempts={attempts}", flush=True)
        except Exception as exc:
            errors += 1
            append_jsonl(err_path, {"task_id": tid, "error": str(exc), "created_at": now()})
            print(f"[{i}] {tid} FAIL {exc}", flush=True)
        if args.delay > 0:
            time.sleep(args.delay)
    materialize(all_path, split, train_path, val_path)
    summary = {
        "created_at": now(),
        "input": str(in_path),
        "output_dir": str(out_dir),
        "model": model,
        "relationship": "task_description_to_compact_contract_json",
        "format": "train_val_messages_only",
        "total_tasks": len(tasks),
        "written_tasks": len(written_ids(all_path)),
        "new_errors": errors,
        "excluded_fields": sorted(EXCLUDED),
        "assistant_contract_fields": list(FIELDS),
        "quality_focus": "concrete_atomic_contracts_not_prompt_repetition",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"\n[DONE] written={summary['written_tasks']} new_errors={errors}")


if __name__ == "__main__":
    main()