from __future__ import annotations

import ast
import copy
import re
from typing import Any

from src.common.task_utils import clean
from src.contract_synthesis import contract_schema_constants as C
from src.contract_synthesis.contract_schema import make_contract_schema_for_task, task_summary

SOURCES = {"signature", "type_hint", "explicit", "example", "strongly_implied", "inferred"}
INVALID_SOURCES = SOURCES | {"not_specified"}
PRE_KINDS = {"domain", "structural", "relational", "format", "membership", "numeric_range"}
POST_KINDS = {"semantic", "relational", "ordering", "membership", "numeric", "structural", "side_effect"}
TARGETS = {"state", "input", "output", "collection_element"}
SOURCE_ALIASES = {"prompt": "explicit"}
PLACEHOLDERS = (
    "short source phrase",
    "parameter_name",
    "valid boundary case",
    "expected behavior supported",
)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def scrub(value: Any) -> str:
    value = clean(value)
    lowered = value.lower()
    return "" if any(marker in lowered for marker in PLACEHOLDERS) else value


def choose(value: Any, allowed: set[str], default: str) -> str:
    value = SOURCE_ALIASES.get(clean(value), clean(value))
    return value if value in allowed else default


def parse_signature(signature: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(f"{clean(signature)}\n    pass")
    except SyntaxError:
        return None

    node = tree.body[0] if tree.body else None
    return node if isinstance(node, ast.FunctionDef) else None


def annotation(node: ast.AST | None) -> str:
    try:
        return ast.unparse(node).replace("typing.", "") if node else ""
    except Exception:
        return ""


def signature_params(signature: str) -> dict[str, str]:
    node = parse_signature(signature)
    if node is None:
        return {}

    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return {arg.arg: annotation(arg.annotation) for arg in args if arg.arg != "self"}


def signature_return(signature: str) -> str:
    node = parse_signature(signature)
    return annotation(node.returns) if node else ""


def normalize_interface(raw: Any, signature: str) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}

    raw_inputs = {
        clean(item.get("name")): item
        for item in as_list(raw.get("inputs"))
        if isinstance(item, dict) and clean(item.get("name"))
    }

    params = signature_params(signature)
    names = list(params) or list(raw_inputs)

    inputs = []
    for name in names:
        item = raw_inputs.get(name, {})
        inputs.append(
            {
                "name": name,
                "type": clean(item.get("type")) or params.get(name, ""),
                "description": scrub(item.get("description")) or f"Input parameter {name}.",
                "source": choose(item.get("source"), SOURCES, "signature"),
            }
        )

    output = raw.get("output") if isinstance(raw.get("output"), dict) else {}

    return {
        "inputs": inputs,
        "output": {
            "type": clean(output.get("type")) or signature_return(signature),
            "description": scrub(output.get("description")) or "Return value produced by the function.",
        },
    }


def clause(target: str, kind: str, description: str, source: str) -> dict[str, str]:
    return {
        "target": target,
        "kind": kind,
        "description": description,
        "source": source,
    }


def interface_types(interface: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: clean(item.get("type"))
        for item in as_list(interface.get("inputs"))
        if isinstance(item, dict) and item.get("name") and clean(item.get("type"))
    }


def infer_target(item: dict[str, Any], names: set[str]) -> str:
    target = clean(item.get("target"))
    if target in names:
        return target

    description = clean(item.get("description"))
    matches = [name for name in names if re.search(rf"\b{re.escape(name)}\b", description)]
    return matches[0] if len(matches) == 1 else ""


def normalize_preconditions(raw: Any, param_types: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for item in as_list(raw):
        if not isinstance(item, dict):
            continue

        description = scrub(item.get("description"))
        target = infer_target(item, set(param_types))

        if not target or not description:
            continue

        rows.append(
            clause(
                target,
                choose(item.get("kind"), PRE_KINDS, "domain"),
                description,
                choose(item.get("source"), SOURCES, "strongly_implied"),
            )
        )

    return rows


def normalize_postconditions(
    raw: Any,
    output_type: str,
    output_source: str,
    output_description: str,
    summary: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fallback = scrub(output_description) or summary

    if output_type and output_type != "None":
        rows.append(
            clause(
                "return",
                "structural",
                f"result must satisfy the return type {output_type}.",
                output_source,
            )
        )

    for item in as_list(raw):
        if not isinstance(item, dict):
            continue

        kind = choose(item.get("kind"), POST_KINDS, "semantic")
        if kind == "type":
            continue

        description = scrub(item.get("description")) or f"result must satisfy: {fallback}"
        rows.append(
            clause(
                "return",
                kind,
                description,
                choose(item.get("source"), SOURCES, "explicit"),
            )
        )

    if not any(row.get("kind") == "semantic" for row in rows):
        rows.append(clause("return", "semantic", f"result must satisfy: {fallback}", "explicit"))

    return rows


def normalize_invariants(raw: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for item in as_list(raw):
        if not isinstance(item, dict):
            continue

        description = scrub(item.get("description"))
        if not description:
            continue

        rows.append(
            {
                "target": choose(item.get("target"), TARGETS, "state"),
                "description": description,
                "source": choose(item.get("source"), SOURCES, "strongly_implied"),
            }
        )

    return rows


def normalize_edge_cases(raw: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for item in as_list(raw):
        if not isinstance(item, dict):
            continue

        case = scrub(item.get("case"))
        expected = scrub(item.get("expected_behavior"))

        if not case and not expected:
            continue

        rows.append(
            {
                "case": case,
                "expected_behavior": expected,
                "source": choose(item.get("source"), SOURCES, "explicit"),
            }
        )

    return rows


def invalid_input_behavior(raw: Any) -> dict[str, Any]:
    default = copy.deepcopy(C.CONTRACT_SCHEMA["invalid_input_behavior"])

    if not isinstance(raw, dict) or raw.get("specified") is not True:
        return default

    return {
        "specified": True,
        "expected_behavior": scrub(raw.get("expected_behavior")) or "not_specified",
        "exception_type": clean(raw.get("exception_type")) or None,
        "description": scrub(raw.get("description")),
        "source": choose(raw.get("source"), INVALID_SOURCES, "explicit"),
    }


def renumber(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    for index, row in enumerate(rows, start=1):
        row["id"] = f"{prefix}{index}"
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, str]] = set()
    result: list[dict[str, Any]] = []

    for row in rows:
        key = (
            row.get("target") or row.get("case"),
            row.get("kind"),
            clean(row.get("description") or row.get("expected_behavior")).lower(),
        )

        if key not in seen:
            seen.add(key)
            result.append(row)

    return result


def normalize_contract(contract: dict[str, Any], task: dict[str, Any], benchmark: str) -> dict[str, Any]:
    base = copy.deepcopy(C.CONTRACT_SCHEMA)
    task_meta = make_contract_schema_for_task(task, benchmark)["task"]
    signature = task_meta["signature"]
    summary = scrub((contract.get("task") or {}).get("summary")) or task_summary(task)

    base["schema_version"] = C.SCHEMA_VERSION
    base["task"] = task_meta
    base["task"]["summary"] = summary
    base["interface"] = normalize_interface(contract.get("interface"), signature)

    signature_types = signature_params(signature)
    input_types = interface_types(base["interface"])
    param_types = {
        name: signature_types.get(name) or input_types.get(name, "")
        for name in (signature_types or input_types)
    }

    output_type = signature_return(signature) or base["interface"]["output"].get("type", "")
    output_source = "signature" if signature_return(signature) else "inferred"

    base["preconditions"] = renumber(
        dedupe(normalize_preconditions(contract.get("preconditions"), param_types)),
        "P",
    )
    base["postconditions"] = renumber(
        dedupe(
            normalize_postconditions(
                contract.get("postconditions"),
                output_type,
                output_source,
                base["interface"]["output"].get("description", ""),
                summary,
            )
        ),
        "Q",
    )
    base["invariants"] = renumber(dedupe(normalize_invariants(contract.get("invariants"))), "I")
    base["edge_cases"] = renumber(dedupe(normalize_edge_cases(contract.get("edge_cases"))), "E")
    base["invalid_input_behavior"] = invalid_input_behavior(contract.get("invalid_input_behavior"))

    return base


__all__ = ["normalize_contract"]