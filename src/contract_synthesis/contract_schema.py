from __future__ import annotations
import copy
import json
import re
from typing import Any
from src.common.task_utils import clean, task_entry_point, task_identifier, task_prompt
from src.contract_synthesis import contract_schema_constants as C

DEF_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(")
IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+[A-Za-z_][\w.]*\s+import\s+.+|import\s+[A-Za-z_][\w.]*.*)\s*$"
)
TYPING_NAMES = {
    "Any",
    "Callable",
    "Deque",
    "Dict",
    "FrozenSet",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Literal",
    "Mapping",
    "MutableMapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
    "Union",
}
SIGNATURE_FIELDS = (
    "signature",
    "declaration",
    "function_signature",
    "entry_point_signature",
)

def first_present(task: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = clean(task.get(field))
        if value:
            return value
    return ""

def explicit_signature(task: dict[str, Any]) -> str:
    return first_present(task, SIGNATURE_FIELDS)

def extract_import_lines(prompt: str) -> list[str]:
    imports: list[str] = []

    for match in IMPORT_RE.finditer(clean(prompt)):
        line = " ".join(match.group(0).split())
        if line and line not in imports:
            imports.append(line)

    return imports

def infer_imports(signature: str) -> list[str]:
    names = sorted(name for name in TYPING_NAMES if re.search(rf"\b{re.escape(name)}\b", signature))
    return [f"from typing import {', '.join(names)}"] if names else []

def infer_imports_from_prompt_and_signature(prompt: str, signature: str) -> list[str]:
    imports = extract_import_lines(prompt)

    for item in infer_imports(signature):
        if item not in imports:
            imports.append(item)

    return imports

def extract_signature(prompt: str, entry_point: str | None) -> str:
    prompt = clean(prompt)
    entry_point = clean(entry_point)

    if not prompt or not entry_point:
        return ""

    match = re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", prompt)
    if not match:
        return ""

    start = match.start()
    index = match.end()
    depth = 1
    quote: str | None = None
    escape = False

    while index < len(prompt) and depth:
        char = prompt[index]

        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
        else:
            if char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

        index += 1

    if depth:
        return ""

    while index < len(prompt) and prompt[index].isspace():
        index += 1

    if prompt.startswith("->", index):
        index += 2
        while index < len(prompt) and prompt[index] not in ":\n\r":
            index += 1

    if index < len(prompt) and prompt[index] == ":":
        return " ".join(prompt[start : index + 1].split())

    return ""

def helper_functions(prompt: str, entry_point: str | None) -> list[str]:
    entry_point = clean(entry_point)
    helpers: list[str] = []

    for name in DEF_RE.findall(clean(prompt)):
        if name != entry_point and name not in helpers:
            helpers.append(name)

    return helpers

def task_summary(task: dict[str, Any]) -> str:
    prompt = task_prompt(task)
    entry_point = task_entry_point(task)

    for marker in ('"""', "'''"):
        parts = prompt.split(marker)
        if len(parts) >= 2:
            sentence = " ".join(parts[1].split()).split(".")[0].strip()
            if sentence:
                return f"{sentence}."

    if entry_point:
        return f"Implement {entry_point} according to the benchmark prompt."

    return "Implement the function according to the benchmark prompt."


def make_contract_schema_for_task(task: dict[str, Any], benchmark: str) -> dict[str, Any]:
    prompt = task_prompt(task)
    entry_point = task_entry_point(task)
    signature = explicit_signature(task) or extract_signature(prompt, entry_point)

    schema = copy.deepcopy(C.CONTRACT_SCHEMA)
    schema["task"].update(
        {
            "task_id": task_identifier(task),
            "benchmark": clean(benchmark),
            "language": "python",
            "entry_point": entry_point,
            "signature": signature,
            "imports_required": infer_imports_from_prompt_and_signature(prompt, signature),
            "helper_functions_required": helper_functions(prompt, entry_point),
            "summary": "",
        }
    )
    return schema

def build_contract_prompt(task: dict[str, Any], benchmark: str) -> str:
    schema = make_contract_schema_for_task(task, benchmark)

    return "\n".join(
        (
            "You are a contract synthesis engine for Python benchmark programming tasks.",
            C.CONTRACT_RULES,
            "",
            "Required JSON object schema:",
            json.dumps(schema, indent=2, ensure_ascii=False),
            "",
            "Fill this schema with a faithful raw contract for the task.",
            "Return exactly one valid JSON object only.",
            "",
            "Task ID:",
            schema["task"]["task_id"],
            "Benchmark:",
            clean(benchmark),
            "Entry point:",
            schema["task"]["entry_point"],
            "Signature:",
            schema["task"]["signature"],
            "Original benchmark prompt:",
            task_prompt(task),
        )
    )

def make_humaneval_contract_schema_for_task(task: dict[str, Any], benchmark: str) -> dict[str, Any]:
    return make_contract_schema_for_task(task, benchmark)

def make_bigcodebench_contract_schema_for_task(task: dict[str, Any], benchmark: str) -> dict[str, Any]:
    return make_contract_schema_for_task(task, benchmark)

__all__ = [
    "build_contract_prompt",
    "extract_signature",
    "helper_functions",
    "infer_imports",
    "make_bigcodebench_contract_schema_for_task",
    "make_contract_schema_for_task",
    "make_humaneval_contract_schema_for_task",
    "task_summary",
]