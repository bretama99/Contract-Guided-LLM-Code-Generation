from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, TypeAlias

JsonDict: TypeAlias = dict[str, Any]

DEPENDENCY_KEYS: Final[tuple[str, ...]] = (
    "reads",
    "modifies",
    "preserves",
    "calls",
    "uses_libraries",
)

_FIELD: Final[JsonDict] = {
    "name": "",
    "description": "",
    "initial_value": "",
    "source": "",
}

_INPUT: Final[JsonDict] = {
    "name": "",
    "type": "",
    "description": "",
}

_OUTPUT: Final[JsonDict] = {
    "type": "",
    "description": "",
}

_INVALID_INPUT_BEHAVIOR: Final[JsonDict] = {
    "specified": False,
    "expected_behavior": "",
    "exception_type": "",
    "description": "",
    "source": "",
}

_INTERACTION: Final[JsonDict] = {
    "name": "",
    "method_sequence": [],
    "preconditions": [],
    "postconditions": [],
    "invariants": [],
    "edge_cases": [],
    "source": "",
}


def new_dependencies_schema() -> dict[str, list[Any]]:
    return {key: [] for key in DEPENDENCY_KEYS}


def new_method_contract_schema() -> JsonDict:
    return {
        "interface": {
            "inputs": [],
            "output": deepcopy(_OUTPUT),
        },
        "preconditions": [],
        "postconditions": [],
        "invariants": [],
        "edge_cases": [],
        "invalid_input_behavior": deepcopy(_INVALID_INPUT_BEHAVIOR),
    }


def new_constructor_schema() -> JsonDict:
    return {
        "signature": "",
        "initializes": [],
        "contract": new_method_contract_schema(),
    }


def new_method_schema() -> JsonDict:
    return {
        "method_name": "",
        "signature": "",
        "is_static": False,
        "dependencies": new_dependencies_schema(),
        "contract": new_method_contract_schema(),
    }


def new_contract_schema() -> JsonDict:
    return {
        "task": {
            "task_id": "",
            "benchmark": "ClassEval",
            "language": "python",
            "execution_model": "class_level",
            "class_name": "",
            "entry_point": "",
            "summary": "",
        },
        "class_interface": {
            "fields": [],
            "methods": [],
        },
        "constructor": new_constructor_schema(),
        "class_invariants": [],
        "method_contracts": [],
        "interaction_contracts": [],
    }


def schema_for_prompt() -> JsonDict:
    schema = new_contract_schema()
    schema["class_interface"]["fields"] = [deepcopy(_FIELD)]
    schema["class_interface"]["methods"] = ["method_name"]
    schema["constructor"]["contract"]["interface"]["inputs"] = [deepcopy(_INPUT)]
    schema["method_contracts"] = [new_method_schema()]
    schema["method_contracts"][0]["contract"]["interface"]["inputs"] = [deepcopy(_INPUT)]
    schema["interaction_contracts"] = [deepcopy(_INTERACTION)]
    return schema


__all__ = [
    "DEPENDENCY_KEYS",
    "JsonDict",
    "new_contract_schema",
    "new_constructor_schema",
    "new_dependencies_schema",
    "new_method_contract_schema",
    "new_method_schema",
    "schema_for_prompt",
]