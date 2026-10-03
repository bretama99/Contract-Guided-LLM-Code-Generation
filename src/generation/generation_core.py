from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


JsonDict = dict[str, Any]

CONTRACT_KEY_ORDER = (
    "interface",
    "preconditions",
    "postconditions",
    "invariants",
)

CONTRACT_KEYS = set(CONTRACT_KEY_ORDER)
CONTRACT_TOP_KEYS = CONTRACT_KEYS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return name or "task"


def output_path(directory: Path, task_id_value: str) -> Path:
    return directory / f"{safe_name(task_id_value)}.json"


def save_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)

        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        return json.loads(text)
    except ValueError:
        pass

    rows = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            rows.append(json.loads(line))
        except ValueError as error:
            raise ValueError(
                f"Invalid JSONL at {path}:{line_number}: {error}"
            ) from error

    return rows


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text:
        return None

    try:
        return json.loads(text)
    except ValueError:
        pass

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def load_tasks(path: Path) -> list[JsonDict]:
    data = read_json(path)

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        for key in ("tasks", "data", "items", "problems", "questions"):
            rows = data.get(key)

            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]

    raise ValueError(f"Could not find task list in {path}")


def selected_tasks(
    tasks: list[JsonDict],
    start: int,
    count: int | None,
) -> list[tuple[int, JsonDict]]:
    if start < 0:
        raise ValueError("start must be non-negative")

    if count is not None and count < 0:
        raise ValueError("count must be non-negative")

    end = None if count is None else start + count
    return list(enumerate(tasks[start:end], start=start))


def task_id(task: JsonDict, index: int) -> str:
    value = task.get("task_id")

    if value not in (None, "", "None"):
        return str(value)

    split = task.get("split")
    split_index = task.get("split_index")

    if split is not None and split_index is not None:
        return f"taco_{split}_{split_index}"

    return f"taco_{index}"


def taco_question(task: JsonDict) -> str:
    value = task.get("question")
    return value.strip() if isinstance(value, str) else ""


def taco_io(task: JsonDict) -> JsonDict:
    value = parse_jsonish(task.get("input_output"))
    return value if isinstance(value, dict) else {}


def taco_function_name(task: JsonDict) -> str | None:
    for key in ("function_name", "fn_name"):
        value = task.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    value = taco_io(task).get("fn_name")

    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def normalize_prompt_text(value: Any) -> str:
    output = []

    for character in str(value or ""):
        if character in ("\u200b", "\ufeff"):
            continue

        if unicodedata.category(character) == "Zs":
            output.append(" ")
        else:
            output.append(character)

    return "".join(output)


def normalize_json_whitespace(value: Any) -> str:
    output = []
    in_string = False
    escaped = False

    for character in str(value or ""):
        if in_string:
            output.append(character)

            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False

            continue

        if character == '"':
            in_string = True
            output.append(character)
        elif character in ("\u200b", "\ufeff"):
            continue
        elif unicodedata.category(character) == "Zs":
            output.append(" ")
        else:
            output.append(character)

    return "".join(output)


def contract_object(value: Any) -> JsonDict | None:
    if isinstance(value, dict) and set(value) == CONTRACT_KEYS:
        return value

    return None


def parse_contract(text: str) -> JsonDict | None:
    try:
        return contract_object(json.loads(text))
    except ValueError:
        return None


def canonical_contract(value: Any) -> JsonDict | None:
    if not isinstance(value, dict):
        return None

    for candidate in (value, value.get("contract")):
        if (
            isinstance(candidate, dict)
            and CONTRACT_KEYS.issubset(candidate)
        ):
            return {
                key: candidate[key]
                for key in CONTRACT_KEY_ORDER
            }

    return None


def parse_json_dict(text: str) -> JsonDict | None:
    try:
        value = json.loads(text)
    except ValueError:
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None

    return value if isinstance(value, dict) else None


def response_after_reasoning(raw: str) -> str:
    text = str(raw or "").strip()

    if text.startswith("<think>"):
        _, separator, final = text.partition("</think>")
        return final.strip() if separator else ""

    boundary = re.search(
        r"(?m)^\s*</think>\s*(?:\n|$)",
        text,
    )

    return text[boundary.end():].strip() if boundary else text


def extract_json_object(
    raw: str,
    *,
    strict: bool = False,
) -> JsonDict | None:
    text = normalize_json_whitespace(raw).strip()

    if not text:
        return None

    def parse(text_value: str) -> JsonDict | None:
        if strict:
            return parse_contract(text_value)

        return canonical_contract(parse_json_dict(text_value))

    select = contract_object if strict else canonical_contract
    contract = parse(text)

    if contract is not None:
        return contract

    if not strict:
        text = response_after_reasoning(text)

        if not text:
            return None

        contract = parse(text)

        if contract is not None:
            return contract

    pattern = (
        r"```(?:json)?\s*(.*?)```"
        if strict
        else r"```(?:json|python|py)?\s*(.*?)```"
    )

    for block in re.findall(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        contract = parse(block.strip())

        if contract is not None:
            return contract

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue

        contract = select(value)

        if contract is not None:
            return contract

    return None