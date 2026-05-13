import copy
import json
import re
from typing import Any
from src.common.task_utils import task_identifier, task_entry_point, task_prompt
from src.contract_synthesis import contract_schema_constants as C

DEF_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(")
TYPING_NAMES = {
    "Any", "Callable", "Deque", "Dict", "FrozenSet", "Generator",
    "Iterable", "Iterator", "List", "Literal", "Mapping",
    "MutableMapping", "Optional", "Sequence", "Set", "Tuple",
    "Type", "Union",
}

BIGCODE_PROMPT_FIELDS = (
    "prompt",
    "complete_prompt",
    "code_prompt",
    "instruction",
    "description",
    "task_description",
)

BIGCODE_SIGNATURE_FIELDS = (
    "signature",
    "declaration",
    "function_signature",
    "entry_point_signature",
)

BIGCODE_ENTRY_POINT_FIELDS = (
    "entry_point",
    "entrypoint",
    "function_name",
    "name",
)

BIGCODE_TASK_ID_FIELDS = (
    "task_id",
    "id",
    "problem_id",
)

IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+[A-Za-z_][\w.]*\s+import\s+.+|import\s+[A-Za-z_][\w.]*.*)\s*$"
)


def first_present(task: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = clean(task.get(field))
        if value:
            return value
    return ""


def bigcode_task_id(task: dict[str, Any]) -> str:
    return first_present(task, BIGCODE_TASK_ID_FIELDS)


def bigcode_entry_point(task: dict[str, Any]) -> str:
    return first_present(task, BIGCODE_ENTRY_POINT_FIELDS)


def bigcode_prompt(task: dict[str, Any]) -> str:
    preferred_fields = (
        "complete_prompt",
        "prompt",
        "code_prompt",
        "instruction",
        "description",
        "task_description",
    )

    candidates = []

    for field in preferred_fields:
        value = clean(task.get(field))
        if value:
            candidates.append(value)

    if not candidates:
        return ""

    return max(candidates, key=len)


def bigcode_explicit_signature(task: dict[str, Any]) -> str:
    return first_present(task, BIGCODE_SIGNATURE_FIELDS)

def explicit_signature(task: dict[str, Any]) -> str:
    return first_present(task, BIGCODE_SIGNATURE_FIELDS) 
def extract_import_lines(prompt: str) -> list[str]:
    imports = []

    for match in IMPORT_RE.finditer(clean(prompt)):
        line = " ".join(match.group(0).split())
        if line and line not in imports:
            imports.append(line)

    return imports


def infer_imports_from_prompt_and_signature(prompt: str, signature: str) -> list[str]:
    imports = extract_import_lines(prompt)
    typing_imports = infer_imports(signature)

    for item in typing_imports:
        if item not in imports:
            imports.append(item)

    return imports

def clean(value: Any) -> str:
    return str(value or "").strip()

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
    while index < len(prompt) and depth:
        char = prompt[index]
        previous = prompt[index - 1] if index else ""
        if quote and char == quote and previous != "\\":
            quote = None
        elif not quote and char in {"'", '"'}:
            quote = char
        elif not quote and char == "(":
            depth += 1
        elif not quote and char == ")":
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
        return " ".join(prompt[start:index + 1].split())
    return ""

def infer_imports(signature: str) -> list[str]:
    names = sorted(
        name
        for name in TYPING_NAMES
        if re.search(rf"\b{re.escape(name)}\b", signature)
    )
    return [f"from typing import {', '.join(names)}"] if names else []

def helper_functions(prompt: str, entry_point: str | None) -> list[str]:
    entry_point = clean(entry_point)
    helpers: list[str] = []
    for name in DEF_RE.findall(clean(prompt)):
        if name != entry_point and name not in helpers:
            helpers.append(name)
    return helpers

def task_summary(task: dict[str, Any]) -> str:
    prompt = bigcode_prompt(task) or clean(task.get("prompt"))
    entry_point = bigcode_entry_point(task) or clean(task.get("entry_point"))

    for marker in ('"""', "'''"):
        parts = prompt.split(marker)
        if len(parts) >= 2:
            sentence = " ".join(parts[1].split()).split(".")[0].strip()
            if sentence:
                return f"{sentence}."

    if entry_point:
        return f"Implement {entry_point} according to the benchmark prompt."

    return "Implement the function according to the benchmark prompt."

def make_humaneval_contract_schema_for_task(
    task: dict[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    prompt = clean(task.get("prompt"))
    entry_point = clean(task.get("entry_point"))
    signature = extract_signature(prompt, entry_point)
    schema = copy.deepcopy(C.CONTRACT_SCHEMA)
    schema["task"].update(
        {
            "task_id": clean(task.get("task_id")),
            "benchmark": clean(benchmark),
            "language": "python",
            "entry_point": entry_point,
            "signature": signature,
            "imports_required": infer_imports(signature),
            "helper_functions_required": helper_functions(prompt, entry_point),
            "summary": "",
        }
    )
    return schema
def make_bigcodebench_contract_schema_for_task(
    task: dict[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    prompt = bigcode_prompt(task)
    entry_point = bigcode_entry_point(task)

    signature = (
        bigcode_explicit_signature(task)
        or extract_signature(prompt, entry_point)
    )

    schema = copy.deepcopy(C.CONTRACT_SCHEMA)
    schema["task"].update(
        {
            "task_id": bigcode_task_id(task),
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

def make_contract_schema_for_task(
    task: dict[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    prompt = bigcode_prompt(task)
    entry_point = bigcode_entry_point(task)

    signature = (
        explicit_signature(task)
        or extract_signature(prompt, entry_point)
    )

    schema = copy.deepcopy(C.CONTRACT_SCHEMA)
    schema["task"].update(
        {
            "task_id": bigcode_task_id(task),
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
    benchmark_name = clean(benchmark).lower()

    prompt = (
        bigcode_prompt(task)
        if benchmark_name == "bigcodebench"
        else clean(task.get("prompt"))
    )

    return "\n".join(
        (
            "You are a contract synthesis engine for Python benchmark programming tasks.",
            C.CONTRACT_RULES,
            "Required JSON object schema:",
            json.dumps(schema, indent=2, ensure_ascii=False),
            "Fill this schema with a faithful raw contract for the task.",
            "Use only the original prompt, visible signature, visible imports, docstring, examples, and visible helper code.",
            "Do not use tests, canonical solutions, generated code, execution feedback, validation results, or repair feedback.",
            "Return exactly one valid JSON object only.",
            "Task ID:",
            schema["task"]["task_id"],
            "Benchmark:",
            clean(benchmark),
            "Entry point:",
            schema["task"]["entry_point"],
            "Signature:",
            schema["task"]["signature"],
            "Original benchmark prompt:",
            prompt,
        )
    )