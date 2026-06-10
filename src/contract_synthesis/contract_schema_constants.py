from __future__ import annotations
from copy import deepcopy
from typing import Any, Final, TypeAlias
JsonDict: TypeAlias = dict[str, Any]

SCHEMA_VERSION: Final[str] = "stage2_raw_contract_schema_v1"
CALLABLE_CONTRACT_SCHEMA: Final[JsonDict] = {
    "interface": {
        "inputs": [],
        "output": {
            "type": "",
            "description": "",
        },
    },
    "preconditions": [],
    "postconditions": [],
    "invariants": [],
    "edge_cases": [],
    "invalid_input_behavior": {
        "specified": False,
        "expected_behavior": "",
        "exception_type": "",
        "description": "",
        "source": "not_specified",
    },
}

CONTRACT_SCHEMA: Final[JsonDict] = {
    "schema_version": SCHEMA_VERSION,
    "task": {
        "task_id": "",
        "benchmark": "",
        "language": "python",
        "entry_point": "",
        "signature": "",
        "imports_required": [],
        "helper_functions_required": [],
        "summary": "",
    },
    **deepcopy(CALLABLE_CONTRACT_SCHEMA),
}

CONTRACT_RULES: Final[str] = """
You generate structured Design-by-Contract specifications for Python functions.
Write all field values in English.

A contract describes:
- preconditions: caller obligations before the function is called
- postconditions: function guarantees after successful execution
- invariants: properties preserved across execution when applicable
- edge_cases: valid but special inputs and expected behavior
- invalid_input_behavior: only behavior explicitly specified for invalid inputs

Return exactly one valid JSON object. No markdown. No explanation.

Allowed source values:
signature | type_hint | explicit | example | strongly_implied | inferred | not_specified

Allowed precondition.kind values:
domain | structural | relational | format | membership | numeric_range

Allowed postcondition.kind values:
semantic | relational | ordering | membership | numeric | structural | side_effect

Allowed invariant.target values:
state | input | output | collection_element

Rules:
- Use only the visible prompt, signature, imports, docstring, examples, and helper code.
- Do not use tests, canonical solutions, generated code, execution feedback, validation results, or repair feedback.
- Do not invent constraints such as non-empty, positive, sorted, unique, finite, or non-null unless clearly supported.
- Preconditions describe valid calling conditions, not implementation checks.
- Postconditions describe observable behavior, not implementation details.
- Every postcondition should be concrete and testable.
- Leave invalid_input_behavior.specified as false unless the prompt explicitly says what to do for invalid inputs.
- Prefer concise, precise clauses over long explanations.
""".strip()

def new_callable_contract_schema() -> JsonDict:
    return deepcopy(CALLABLE_CONTRACT_SCHEMA)

def new_contract_schema() -> JsonDict:
    return deepcopy(CONTRACT_SCHEMA)

__all__ = [
    "CALLABLE_CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA",
    "CONTRACT_RULES",
    "SCHEMA_VERSION",
    "new_callable_contract_schema",
    "new_contract_schema",
]