from __future__ import annotations
from copy import deepcopy
from typing import Any, Final, TypeAlias
from src.contract_synthesis.contract_schema_constants import (
    new_callable_contract_schema,
)
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
_METHOD_INTERFACE: Final[JsonDict] = {
    "name": "",
    "signature": "",
    "description": "",
}
_INTERACTION_CONTRACT: Final[JsonDict] = {
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
    return new_callable_contract_schema()

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
    schema["class_interface"]["methods"] = [deepcopy(_METHOD_INTERFACE)]
    schema["method_contracts"] = [new_method_schema()]
    schema["interaction_contracts"] = [deepcopy(_INTERACTION_CONTRACT)]
    return schema

__all__ = [
    "DEPENDENCY_KEYS",
    "new_contract_schema",
    "new_constructor_schema",
    "new_dependencies_schema",
    "new_method_contract_schema",
    "new_method_schema",
    "schema_for_prompt",
]