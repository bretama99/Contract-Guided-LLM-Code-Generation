from __future__ import annotations

import re
from typing import Any

from src.classeval.core import JsonDict, as_dict, as_list, text


_PARAM_RE = re.compile(r":param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)", re.I)
_RETURN_RE = re.compile(r":return:\s*(.+)", re.I)
_DEF_RE = re.compile(r"^\s*(?:@staticmethod\s*)?\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*:", re.M)

_PRECONDITION_CUES = (
    "should be",
    "must be",
    "has to be",
    "greater than",
    "less than",
    "at least",
    "at most",
    "positive",
    "negative",
    "non-negative",
    "non empty",
    "non-empty",
    "valid",
    "exists",
    "already",
    "in the",
    "from the",
    "not empty",
    "between",
    "integer",
    "float",
    "string",
    "list",
    "dict",
    "dictionary",
)

_EDGE_CUES = (
    "empty",
    "missing",
    "not found",
    "does not exist",
    "already exists",
    "duplicate",
    "out of stock",
    "insufficient",
    "invalid",
    "zero",
    "negative",
    "boundary",
    "maximum",
    "minimum",
    "greater than",
    "less than",
)

_MUTATION_CUES = (
    "add",
    "remove",
    "delete",
    "update",
    "set",
    "insert",
    "append",
    "withdraw",
    "deposit",
    "purchase",
    "restock",
    "replenish",
    "clear",
    "reset",
    "modify",
    "change",
)


def _clean_clause(value: str) -> str:
    value = re.sub(r"\s+", " ", text(value)).strip()
    value = value.strip("-•:;. ")
    return value


def _unique(values: list[str], limit: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _clean_clause(value)
        if not value:
            continue
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
        if len(out) >= limit:
            break
    return out


def _method_doc(info: JsonDict) -> str:
    return text(info.get("method_description")) or text(info.get("description"))


def _method_name(info: JsonDict) -> str:
    return text(info.get("method_name"))


def _signature_from_doc(doc: str, fallback: str = "") -> str:
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line
    return fallback


def _param_docs(doc: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _PARAM_RE.finditer(doc):
        name = match.group(1).strip()
        desc = _clean_clause(match.group(2))
        if name and desc:
            result[name] = desc
    return result


def _return_doc(doc: str) -> str:
    match = _RETURN_RE.search(doc)
    return _clean_clause(match.group(1)) if match else ""


def _dependencies(info: JsonDict) -> JsonDict:
    deps = as_dict(info.get("dependencies"))
    return {
        "reads": as_list(deps.get("field_dependencies")) or as_list(deps.get("reads")),
        "modifies": as_list(deps.get("modifies")),
        "preserves": as_list(deps.get("preserves")),
        "calls": as_list(deps.get("method_dependencies")) or as_list(deps.get("calls")),
        "uses_libraries": as_list(deps.get("lib_dependencies")) or as_list(deps.get("uses_libraries")),
    }


def _looks_static(doc: str) -> bool:
    return "@staticmethod" in doc


def _signature_args(signature: str) -> list[str]:
    match = re.search(r"\((.*?)\)", signature)
    if not match:
        return []
    args = []
    for raw in match.group(1).split(","):
        name = raw.strip().split(":", 1)[0].split("=", 1)[0].strip()
        if name and name not in {"self", "cls"}:
            args.append(name)
    return args


def infer_clause_hints(task: JsonDict) -> JsonDict:
    fields = [text(x) for x in as_list(task.get("fields")) if text(x)]
    constructor = text(task.get("class_constructor"))
    class_description = text(task.get("class_description"))

    class_invariants: list[str] = []
    if fields:
        class_invariants.append("The object maintains initialized fields: " + ", ".join(fields) + ".")
    for field in fields:
        lower = field.lower()
        if any(token in lower for token in ("list", "items", "records", "history", "inventory")):
            class_invariants.append(f"{field} remains a collection storing the class state.")
        elif any(token in lower for token in ("dict", "map", "table", "database", "cache")):
            class_invariants.append(f"{field} remains a mapping storing named class state.")
        elif any(token in lower for token in ("balance", "count", "total", "score", "quantity", "size")):
            class_invariants.append(f"{field} remains the numeric state used by related methods.")

    method_hints: list[JsonDict] = []

    for info in as_list(task.get("methods_info")):
        if not isinstance(info, dict):
            continue

        name = _method_name(info)
        doc = _method_doc(info)
        signature = _signature_from_doc(doc)
        param_docs = _param_docs(doc)
        return_doc = _return_doc(doc)
        args = _signature_args(signature)
        deps = _dependencies(info)

        preconditions: list[str] = []
        postconditions: list[str] = []
        invariants: list[str] = []
        edge_cases: list[str] = []

        for arg in args:
            desc = param_docs.get(arg, "")
            lower = desc.lower()
            if desc and any(cue in lower for cue in _PRECONDITION_CUES):
                preconditions.append(f"{arg}: {desc}")

        for line in doc.splitlines():
            cleaned = _clean_clause(line)
            lower = cleaned.lower()
            if not cleaned or cleaned.startswith(("def ", ":param", ":return", ">>>")):
                continue
            if any(cue in lower for cue in _PRECONDITION_CUES) and any(arg in cleaned for arg in args):
                preconditions.append(cleaned)
            if any(cue in lower for cue in _EDGE_CUES):
                edge_cases.append(cleaned)

        if return_doc:
            postconditions.append(return_doc)

        summary = _clean_clause(doc.replace('"""', "").split("\n", 1)[0])
        if summary and not summary.startswith("def "):
            postconditions.append(summary)

        read_fields = [text(x) for x in as_list(deps.get("reads")) if text(x)]
        modified_fields = [text(x) for x in as_list(deps.get("modifies")) if text(x)]

        if read_fields:
            preconditions.append("Required class state is initialized before use: " + ", ".join(read_fields) + ".")
            invariants.append("Class state remains valid for fields: " + ", ".join(read_fields) + ".")

        if modified_fields:
            invariants.append("Mutating this method preserves valid class state for: " + ", ".join(modified_fields) + ".")

        method_lower = name.lower() + " " + doc.lower()
        if any(cue in method_lower for cue in _MUTATION_CUES) and fields:
            invariants.append("After this method, related object fields remain internally consistent.")

        if not edge_cases:
            for arg, desc in param_docs.items():
                lower = desc.lower()
                if any(cue in lower for cue in _EDGE_CUES):
                    edge_cases.append(f"{arg}: {desc}")

        method_hints.append(
            {
                "method_name": name,
                "signature": signature,
                "is_static": _looks_static(doc),
                "precondition_hints": _unique(preconditions, 6),
                "postcondition_hints": _unique(postconditions, 6),
                "invariant_hints": _unique(invariants, 4),
                "edge_case_hints": _unique(edge_cases, 5),
                "dependencies": deps,
            }
        )

    return {
        "class_invariant_hints": _unique(class_invariants, 8),
        "constructor_hints": {
            "preconditions": [],
            "postconditions": _unique(
                [
                    "The constructor initializes the class state described by the skeleton.",
                    "Initialized fields are available to all instance methods.",
                ],
                4,
            ),
            "invariants": _unique(class_invariants, 6),
            "edge_cases": [],
            "constructor_text": constructor,
            "class_description": class_description,
        },
        "method_clause_hints": method_hints,
    }


def contract_clause_errors(contract: JsonDict, task: JsonDict) -> list[str]:
    errors: list[str] = []
    hints = infer_clause_hints(task)
    method_hint_map = {
        text(item.get("method_name")): item
        for item in as_list(hints.get("method_clause_hints"))
        if isinstance(item, dict)
    }

    class_invariants = as_list(contract.get("class_invariants"))
    if as_list(hints.get("class_invariant_hints")) and not class_invariants:
        errors.append("missing class_invariants despite constructor/field hints")

    for method in as_list(contract.get("method_contracts")):
        if not isinstance(method, dict):
            continue

        name = text(method.get("method_name"))
        body = as_dict(method.get("contract"))
        method_hints = as_dict(method_hint_map.get(name))

        pre = as_list(body.get("preconditions"))
        post = as_list(body.get("postconditions"))
        inv = as_list(body.get("invariants"))
        edge = as_list(body.get("edge_cases"))

        if not post:
            errors.append(f"{name}: missing postconditions")

        if as_list(method_hints.get("precondition_hints")) and not pre:
            errors.append(f"{name}: missing task-specific preconditions")

        if as_list(method_hints.get("invariant_hints")) and not inv:
            errors.append(f"{name}: missing task-specific invariants")

        if as_list(method_hints.get("edge_case_hints")) and not edge:
            errors.append(f"{name}: missing task-specific edge_cases")

        if not pre and not inv and not edge and post:
            errors.append(f"{name}: collapsed into postconditions only")

    return errors


def merge_missing_clause_hints(contract: JsonDict, task: JsonDict) -> JsonDict:
    hints = infer_clause_hints(task)
    method_hint_map = {
        text(item.get("method_name")): item
        for item in as_list(hints.get("method_clause_hints"))
        if isinstance(item, dict)
    }

    if not as_list(contract.get("class_invariants")):
        contract["class_invariants"] = as_list(hints.get("class_invariant_hints"))

    for method in as_list(contract.get("method_contracts")):
        if not isinstance(method, dict):
            continue

        name = text(method.get("method_name"))
        method_hints = as_dict(method_hint_map.get(name))
        body = as_dict(method.get("contract"))
        method["contract"] = body

        if not as_list(body.get("preconditions")):
            body["preconditions"] = as_list(method_hints.get("precondition_hints"))

        if not as_list(body.get("invariants")):
            body["invariants"] = as_list(method_hints.get("invariant_hints"))

        if not as_list(body.get("edge_cases")):
            body["edge_cases"] = as_list(method_hints.get("edge_case_hints"))

        if not as_list(body.get("postconditions")):
            body["postconditions"] = as_list(method_hints.get("postcondition_hints"))

    return contract