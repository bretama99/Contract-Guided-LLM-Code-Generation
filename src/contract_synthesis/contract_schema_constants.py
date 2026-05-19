from typing import Any, Final


SCHEMA_VERSION: Final[str] = "2.9"
CREATED_STAGE: Final[str] = "2A_raw_contract_synthesis"
INITIAL_STATUS: Final[str] = "raw_unvalidated"


CONTRACT_SCHEMA: Final[dict[str, Any]] = {
    "schema_version": SCHEMA_VERSION,
    "lifecycle": {
        "created_stage": CREATED_STAGE,
        "status": INITIAL_STATUS,
    },
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
        "expected_behavior": "not_specified",
        "exception_type": None,
        "description": "",
        "source": "not_specified",
    },
}


CONTRACT_RULES: Final[str] = """
You generate structured Design-by-Contract specifications for Python functions.
Write all field values in English.

A contract describes:
  - preconditions: what must be true before the function is called
  - postconditions: what must be true after successful execution
  - invariants: what must remain true when applicable

Return exactly one valid JSON object. No markdown. No explanation.

---
ALLOWED VALUES
---

source:
  signature | type_hint | explicit | example | strongly_implied | inferred

precondition.kind:
  domain | structural | relational | format | membership | numeric_range

postcondition.kind:
  semantic | relational | ordering | membership | numeric | structural | side_effect

invariant.target:
  state | input | output | collection_element

invalid_input_behavior.source:
  explicit | example | strongly_implied | not_specified

---
FIELD RULES
---

task.summary:
  Write one sentence describing what the function computes.
  Do not write generic text like "Implement the function."

interface.inputs:
  For each parameter, write:
    - name: the parameter name
    - type: the parameter type, from the signature or docstring
    - description: what this parameter means or controls in the task

interface.output:
  Write:
    - type: the return type, from the signature or docstring
    - description: what the returned value means in the task

preconditions:
  Conditions that must be true before the function is called for valid execution.
  They describe caller obligations.
  Include only constraints stated or clearly required by the prompt, signature, or examples.
  Do not invent constraints such as non-empty, positive, sorted, or unique unless clearly supported.

postconditions:
  Conditions that must be true after the function successfully returns, assuming the preconditions hold.
  They describe the function's guarantees.
  Describe the relationship between the output and the inputs.
  Include visible side effects only if the prompt requires them.
  Do not include implementation details.
  Every postcondition must be observable and testable.

edge_cases:
  Valid but special input situations that may affect the result.
  Examples: empty collection if allowed, one-element collection, zero, duplicates, equal values, boundary values, already sorted input, or no matching element.
  Each edge case must describe the input situation and the expected behavior.
  Do not include invalid inputs here; invalid inputs belong in invalid_input_behavior.

invariants:
  Properties that must remain true before and after execution, or across object/state changes.
  Add an invariant only when the task involves persistent state, mutation, preserved structure, or an always-maintained property.

invalid_input_behavior:
  Behavior when inputs violate the valid calling conditions.
  Set specified to true only if the prompt or examples explicitly define what should happen.
  Examples: raise ValueError, return -1, return None.
  If invalid-input behavior is not specified, leave specified as false.

---
QUALITY REQUIREMENTS
---

- The contract must describe behavior, not implementation.
- Preconditions describe caller obligations.
- Postconditions describe supplier/function guarantees.
- Invariants describe always-maintained properties.
- Do not invent constraints.
- Do not add vague postconditions like "returns the correct result."
- Prefer precise, testable statements.
- Use source to show where each claim came from.
- If a claim is only inferred, mark it as inferred.
- Raw contracts may be incomplete; do not over-specify.
""".strip()