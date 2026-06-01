from __future__ import annotations

import ast
import json
from typing import Any

from src.classeval.contract_schema import (
    DEPENDENCY_KEYS,
    JsonDict,
    new_constructor_schema,
    new_contract_schema,
    new_dependencies_schema,
    new_method_contract_schema,
    new_method_schema,
)


def normalize_contract_shape(value: JsonDict) -> JsonDict:
    raw = _dict(value)
    schema = new_contract_schema()
    schema.update(raw)

    return {
        "task": _dict(schema.get("task")),
        "class_interface": _normalize_class_interface(schema.get("class_interface")),
        "constructor": _normalize_constructor(schema.get("constructor")),
        "class_invariants": _clause_list(schema.get("class_invariants")),
        "method_contracts": [
            _normalize_method(item)
            for item in _list(schema.get("method_contracts"))
            if isinstance(item, dict)
        ],
        "interaction_contracts": [
            _normalize_interaction(item)
            for item in _list(schema.get("interaction_contracts"))
            if isinstance(item, dict)
        ],
    }


def stage2_shape_errors(contract: JsonDict, expected_methods: list[str] | None = None) -> list[str]:
    if not isinstance(contract, dict):
        return ["contract must be a JSON object"]

    errors: list[str] = []

    for key in ("task", "class_interface", "constructor"):
        _require_object(contract, key, errors)

    for key in ("class_invariants", "method_contracts", "interaction_contracts"):
        _require_list(contract, key, errors)

    task = _dict(contract.get("task"))
    for key in ("task_id", "class_name", "entry_point"):
        if not _text(task.get(key)):
            errors.append(f"task.{key} is missing")

    class_interface = _dict(contract.get("class_interface"))
    for key in ("fields", "methods"):
        if not isinstance(class_interface.get(key), list):
            errors.append(f"class_interface.{key} must be a list")

    constructor = _dict(contract.get("constructor"))
    if isinstance(constructor.get("contract"), dict):
        _check_callable_contract(constructor["contract"], "constructor.contract", errors)
    else:
        errors.append("constructor.contract must be an object")

    seen: set[str] = set()

    for index, method in enumerate(_list(contract.get("method_contracts"))):
        if not isinstance(method, dict):
            errors.append(f"method_contracts[{index}] must be an object")
            continue

        where = f"method_contracts[{index}]"
        name = _text(method.get("method_name"))

        if name:
            seen.add(name)
        else:
            errors.append(f"{where}.method_name is missing")

        if not _text(method.get("signature")):
            errors.append(f"{where}.signature is missing")

        if not isinstance(method.get("is_static"), bool):
            errors.append(f"{where}.is_static must be boolean")

        deps = method.get("dependencies")
        if not isinstance(deps, dict):
            errors.append(f"{where}.dependencies must be an object")
        else:
            for key in DEPENDENCY_KEYS:
                if not isinstance(deps.get(key), list):
                    errors.append(f"{where}.dependencies.{key} must be a list")

        body = method.get("contract")
        if isinstance(body, dict):
            _check_callable_contract(body, f"{where}.contract", errors)
        else:
            errors.append(f"{where}.contract must be an object")

    for name in expected_methods or []:
        if name not in seen:
            errors.append(f"missing method contract: {name}")

    return errors


def _normalize_class_interface(value: Any) -> JsonDict:
    raw = _dict(value)
    return {
        "fields": _list(raw.get("fields")),
        "methods": _list(raw.get("methods")),
    }


def _normalize_constructor(value: Any) -> JsonDict:
    schema = new_constructor_schema()
    schema.update(_dict(value))
    return {
        "signature": _text(schema.get("signature")),
        "initializes": _list(schema.get("initializes")),
        "contract": _normalize_callable_contract(schema.get("contract")),
    }


def _normalize_method(value: JsonDict) -> JsonDict:
    schema = new_method_schema()
    schema.update(value)
    return {
        "method_name": _text(schema.get("method_name")),
        "signature": _text(schema.get("signature")),
        "is_static": _bool(schema.get("is_static")),
        "dependencies": _normalize_dependencies(schema.get("dependencies")),
        "contract": _normalize_callable_contract(_contract_source(schema)),
    }


def _normalize_interaction(value: JsonDict) -> JsonDict:
    return {
        "name": _text(value.get("name")),
        "method_sequence": _list(value.get("method_sequence")),
        "preconditions": _clause_list(value.get("preconditions")),
        "postconditions": _clause_list(value.get("postconditions")),
        "invariants": _clause_list(value.get("invariants")),
        "edge_cases": _clause_list(value.get("edge_cases")),
        "source": _text(value.get("source")),
    }


def _contract_source(method: JsonDict) -> Any:
    contract = _literal(method.get("contract"))
    if isinstance(contract, dict):
        return contract

    lifted = {
        key: method.get(key)
        for key in (
            "interface",
            "preconditions",
            "postconditions",
            "invariants",
            "edge_cases",
            "invalid_input_behavior",
        )
    }
    return lifted if any(value not in (None, "", [], {}) for value in lifted.values()) else contract


def _normalize_callable_contract(value: Any) -> JsonDict:
    schema = new_method_contract_schema()
    raw = _literal(value)

    if isinstance(raw, dict):
        schema.update(raw)
    elif not _is_empty(raw):
        schema["interface"]["output"]["description"] = _text(raw)

    return {
        "interface": _normalize_interface(schema.get("interface")),
        "preconditions": _clause_list(schema.get("preconditions")),
        "postconditions": _clause_list(schema.get("postconditions")),
        "invariants": _clause_list(schema.get("invariants")),
        "edge_cases": _clause_list(schema.get("edge_cases")),
        "invalid_input_behavior": _normalize_invalid_input(schema.get("invalid_input_behavior")),
    }


def _normalize_interface(value: Any) -> JsonDict:
    raw = _literal(value)

    if not isinstance(raw, dict):
        return {
            "inputs": [],
            "output": {
                "type": "",
                "description": "" if _is_empty(raw) else _text(raw),
            },
        }

    output = _dict(raw.get("output"))

    return {
        "inputs": _list(raw.get("inputs")),
        "output": {
            "type": _text(output.get("type")),
            "description": _text(output.get("description")),
        },
    }


def _normalize_invalid_input(value: Any) -> JsonDict:
    raw = _literal(value)

    if not isinstance(raw, dict):
        return {
            "specified": False,
            "expected_behavior": "",
            "exception_type": "",
            "description": "" if _is_empty(raw) else _text(raw),
            "source": "",
        }

    return {
        "specified": _bool(raw.get("specified")),
        "expected_behavior": _text(raw.get("expected_behavior")),
        "exception_type": _text(raw.get("exception_type")),
        "description": _text(raw.get("description")),
        "source": _text(raw.get("source")),
    }


def _normalize_dependencies(value: Any) -> dict[str, list[Any]]:
    raw = _literal(value)
    if isinstance(raw, dict):
        return {key: _list(raw.get(key)) for key in DEPENDENCY_KEYS}
    return new_dependencies_schema()


def _check_callable_contract(contract: JsonDict, where: str, errors: list[str]) -> None:
    _require_object(contract, "interface", errors, where)
    _require_object(contract, "invalid_input_behavior", errors, where)

    interface = _dict(contract.get("interface"))

    if not isinstance(interface.get("inputs"), list):
        errors.append(f"{where}.interface.inputs must be a list")

    if not isinstance(interface.get("output"), dict):
        errors.append(f"{where}.interface.output must be an object")

    for key in ("preconditions", "postconditions", "invariants", "edge_cases"):
        _require_list(contract, key, errors, where)


def _require_object(data: JsonDict, key: str, errors: list[str], prefix: str | None = None) -> None:
    if not isinstance(data.get(key), dict):
        errors.append(f"{prefix + '.' if prefix else ''}{key} must be an object")


def _require_list(data: JsonDict, key: str, errors: list[str], prefix: str | None = None) -> None:
    if not isinstance(data.get(key), list):
        errors.append(f"{prefix + '.' if prefix else ''}{key} must be a list")


def _clause_list(value: Any) -> list[Any]:
    raw = _literal(value)
    return [] if _is_empty(raw) else raw if isinstance(raw, list) else [raw]


def _list(value: Any) -> list[Any]:
    raw = _literal(value)

    if isinstance(raw, list):
        return raw

    if _is_empty(raw) or isinstance(raw, dict):
        return []

    if isinstance(raw, str) and "," in raw:
        return [item.strip().strip("'\"") for item in raw.split(",") if not _is_empty(item)]

    return [raw]


def _dict(value: Any) -> JsonDict:
    raw = _literal(value)
    return raw if isinstance(raw, dict) else {}


def _literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped:
        return ""

    if stripped.lower() in {"none", "null"}:
        return None

    if stripped[0] not in "[{(":
        return stripped

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue

    return stripped


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}

    return bool(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_empty(value: Any) -> bool:
    if value in (None, "", [], {}, ()):
        return True

    if not isinstance(value, str):
        return False

    return value.strip().lower() in {
        "",
        "none",
        "null",
        "[]",
        "{}",
        "not_specified",
        "not specified",
        "not applicable",
        "n/a",
        "nothing",
        "no",
        "no libraries",
    }


__all__ = [
    "normalize_contract_shape",
    "stage2_shape_errors",
]