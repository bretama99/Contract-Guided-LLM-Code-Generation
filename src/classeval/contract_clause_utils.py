from __future__ import annotations

import re
from typing import Any

from src.classeval.core import JsonDict, as_dict, as_list, text

_PARAM_RE = re.compile(r":param\s+([A-Za-z_]\w*)\s*:\s*(.+)", re.I)
_RETURN_RE = re.compile(r":return:\s*(.+)", re.I)

PRE_CUES = (
    "must", "should", "valid", "positive", "negative", "non-negative",
    "non-empty", "not empty", "integer", "float", "string", "list", "dict",
    "greater than", "less than", "at least", "at most", "between", "exists",
)

EDGE_CUES = (
    "empty", "missing", "not found", "does not exist", "already exists",
    "duplicate", "invalid", "zero", "negative", "minimum", "maximum",
    "boundary", "insufficient", "out of stock", "none", "null",
)

MUTATION_CUES = (
    "add", "remove", "delete", "update", "set", "insert", "append",
    "withdraw", "deposit", "purchase", "restock", "clear", "reset",
    "modify", "change",
)

BAD_TEXT = (
    "solution", "reference", "ground truth", "oracle", "self.assert",
    "unittest", "test case", "expected",
)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", text(value)).strip().strip("-•:;. ")


def unique(values: list[Any], limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean(value)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def field_name(value: Any) -> str:
    if isinstance(value, dict):
        return text(value.get("name")) or text(value.get("field")) or text(value.get("description"))
    return text(value)


def fields_of(task: JsonDict) -> list[str]:
    return unique([field_name(item) for item in as_list(task.get("fields"))], 20)


def method_doc(info: JsonDict) -> str:
    return text(info.get("method_description")) or text(info.get("description"))


def method_name(info: JsonDict) -> str:
    return text(info.get("method_name"))


def signature_from_doc(doc: str) -> str:
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line
    return ""


def signature_args(signature: str) -> list[str]:
    match = re.search(r"\((.*?)\)", signature)
    if not match:
        return []

    args: list[str] = []
    for raw in match.group(1).split(","):
        name = raw.strip().split(":", 1)[0].split("=", 1)[0].strip()
        if name and name not in {"self", "cls"}:
            args.append(name)
    return args


def param_docs(doc: str) -> dict[str, str]:
    return {
        match.group(1).strip(): clean(match.group(2))
        for match in _PARAM_RE.finditer(doc)
        if clean(match.group(2))
    }


def return_doc(doc: str) -> str:
    match = _RETURN_RE.search(doc)
    return clean(match.group(1)) if match else ""


def dependencies(info: JsonDict) -> JsonDict:
    deps = as_dict(info.get("dependencies"))
    return {
        "reads": as_list(deps.get("field_dependencies")) or as_list(deps.get("reads")),
        "modifies": as_list(deps.get("modifies")),
        "preserves": as_list(deps.get("preserves")),
        "calls": as_list(deps.get("method_dependencies")) or as_list(deps.get("calls")),
        "uses_libraries": as_list(deps.get("lib_dependencies")) or as_list(deps.get("uses_libraries")),
    }


def safe_doc_lines(doc: str) -> list[str]:
    lines: list[str] = []
    for raw in doc.replace('"""', "").replace("'''", "").splitlines():
        line = clean(raw)
        lower = line.lower()
        if not line or line.startswith(("def ", ":param", ":return:", ">>>")):
            continue
        if any(bad in lower for bad in BAD_TEXT):
            continue
        lines.append(line)
    return lines


def doc_summary(doc: str, name: str) -> str:
    lines = safe_doc_lines(doc)
    return lines[0] if lines else f"Implements the visible behavior of method {name}."


def append_if_cued(target: list[str], value: str, cues: tuple[str, ...]) -> None:
    lower = value.lower()
    if any(cue in lower for cue in cues):
        target.append(value)


def field_invariants(fields: list[str]) -> list[str]:
    invariants: list[str] = []

    if fields:
        invariants.append("The object maintains initialized fields: " + ", ".join(fields) + ".")

    for field in fields:
        lower = field.lower()
        if any(key in lower for key in ("list", "items", "records", "history", "inventory", "queue", "stack")):
            invariants.append(f"{field} remains a collection representing class state.")
        elif any(key in lower for key in ("dict", "map", "table", "database", "cache")):
            invariants.append(f"{field} remains a mapping representing class state.")
        elif any(key in lower for key in ("balance", "count", "total", "score", "quantity", "size")):
            invariants.append(f"{field} remains numeric state used by related methods.")

    return unique(invariants, 10)


def method_hints(info: JsonDict, fields: list[str]) -> JsonDict:
    name = method_name(info)
    doc = method_doc(info)
    signature = signature_from_doc(doc)
    args = signature_args(signature)
    params = param_docs(doc)
    ret = return_doc(doc)
    deps = dependencies(info)

    pre: list[str] = []
    post: list[str] = []
    inv: list[str] = []
    edge: list[str] = []

    for arg in args:
        desc = params.get(arg, "")
        lower = desc.lower()
        if desc and any(cue in lower for cue in PRE_CUES):
            pre.append(f"{arg}: {desc}")
        if desc and any(cue in lower for cue in EDGE_CUES):
            edge.append(f"{arg}: {desc}")

    for line in safe_doc_lines(doc):
        append_if_cued(edge, line, EDGE_CUES)
        if args and any(arg in line for arg in args):
            append_if_cued(pre, line, PRE_CUES)

    if ret:
        post.append(ret)

    post.append(doc_summary(doc, name))

    reads = [text(item) for item in as_list(deps.get("reads")) if text(item)]
    modifies = [text(item) for item in as_list(deps.get("modifies")) if text(item)]
    preserves = [text(item) for item in as_list(deps.get("preserves")) if text(item)]

    if reads:
        pre.append("Required class state is initialized before use: " + ", ".join(reads) + ".")
        inv.append("Class state remains valid for fields read by this method: " + ", ".join(reads) + ".")

    if modifies:
        inv.append("After execution, modified fields remain internally consistent: " + ", ".join(modifies) + ".")

    if preserves:
        inv.append("The method preserves valid state for: " + ", ".join(preserves) + ".")

    if fields and any(cue in f"{name} {doc}".lower() for cue in MUTATION_CUES):
        inv.append("After this method, related object fields remain internally consistent.")

    return {
        "method_name": name,
        "signature": signature,
        "is_static": "@staticmethod" in doc,
        "precondition_hints": unique(pre, 6),
        "postcondition_hints": unique(post, 8),
        "invariant_hints": unique(inv, 6),
        "edge_case_hints": unique(edge, 6),
        "dependencies": deps,
    }


def infer_clause_hints(task: JsonDict) -> JsonDict:
    fields = fields_of(task)
    invariants = field_invariants(fields)

    return {
        "class_invariant_hints": invariants,
        "constructor_hints": {
            "preconditions": [],
            "postconditions": [
                "The constructor initializes the class state described by the skeleton.",
                "Initialized fields are available to all instance methods.",
            ],
            "invariants": invariants,
            "edge_cases": [],
            "constructor_text": text(task.get("class_constructor")),
            "class_description": text(task.get("class_description")),
        },
        "method_clause_hints": [
            method_hints(info, fields)
            for info in as_list(task.get("methods_info"))
            if isinstance(info, dict)
        ],
    }


def contract_clause_errors(contract: JsonDict, task: JsonDict) -> list[str]:
    hints = infer_clause_hints(task)
    hint_by_method = {
        text(item.get("method_name")): item
        for item in as_list(hints.get("method_clause_hints"))
        if isinstance(item, dict)
    }

    warnings: list[str] = []

    if as_list(hints.get("class_invariant_hints")) and not as_list(contract.get("class_invariants")):
        warnings.append("missing class_invariants despite visible field/state hints")

    for method in as_list(contract.get("method_contracts")):
        if not isinstance(method, dict):
            continue

        name = text(method.get("method_name"))
        body = as_dict(method.get("contract"))
        method_hints = as_dict(hint_by_method.get(name))

        groups = {
            "preconditions": as_list(body.get("preconditions")),
            "postconditions": as_list(body.get("postconditions")),
            "invariants": as_list(body.get("invariants")),
            "edge_cases": as_list(body.get("edge_cases")),
        }

        if not any(groups.values()):
            warnings.append(f"{name}: all contract clause groups are empty")

        for group, hint_key in (
            ("preconditions", "precondition_hints"),
            ("postconditions", "postcondition_hints"),
            ("invariants", "invariant_hints"),
            ("edge_cases", "edge_case_hints"),
        ):
            if as_list(method_hints.get(hint_key)) and not groups[group]:
                warnings.append(f"{name}: missing visible {group} hints")

    return warnings


def merge_missing_clause_hints(contract: JsonDict, task: JsonDict) -> JsonDict:
    hints = infer_clause_hints(task)
    hint_by_method = {
        text(item.get("method_name")): item
        for item in as_list(hints.get("method_clause_hints"))
        if isinstance(item, dict)
    }

    if not as_list(contract.get("class_invariants")):
        contract["class_invariants"] = as_list(hints.get("class_invariant_hints"))

    constructor = as_dict(contract.get("constructor"))
    constructor_body = as_dict(constructor.get("contract"))
    constructor_hints = as_dict(hints.get("constructor_hints"))

    if constructor_body:
        constructor_body.setdefault("postconditions", as_list(constructor_hints.get("postconditions")))
        constructor_body.setdefault("invariants", as_list(constructor_hints.get("invariants")))

        if not as_list(constructor_body.get("postconditions")):
            constructor_body["postconditions"] = as_list(constructor_hints.get("postconditions"))
        if not as_list(constructor_body.get("invariants")):
            constructor_body["invariants"] = as_list(constructor_hints.get("invariants"))

        constructor["contract"] = constructor_body
        contract["constructor"] = constructor

    for method in as_list(contract.get("method_contracts")):
        if not isinstance(method, dict):
            continue

        name = text(method.get("method_name"))
        method_hints = as_dict(hint_by_method.get(name))
        body = as_dict(method.get("contract"))
        method["contract"] = body

        for group, hint_key in (
            ("preconditions", "precondition_hints"),
            ("postconditions", "postcondition_hints"),
            ("invariants", "invariant_hints"),
            ("edge_cases", "edge_case_hints"),
        ):
            if not as_list(body.get(group)):
                body[group] = as_list(method_hints.get(hint_key))

    return contract


__all__ = [
    "contract_clause_errors",
    "infer_clause_hints",
    "merge_missing_clause_hints",
]