from typing import Any, Final

SCHEMA_VERSION: Final[str] = "2.9"
CREATED_STAGE: Final[str] = "2A_raw_contract_synthesis"
INITIAL_STATUS: Final[str] = "raw_unvalidated"

CONTRACT_SCHEMA: dict[str, Any] = {
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


from typing import Final

from typing import Final

CONTRACT_RULES: Final[str] = """
You generate structured contracts that precisely describe Python function behaviour.
Write all field values in English.

A contract is a JSON object with these fields:
  task.summary, interface, preconditions, postconditions,
  edge_cases, invariants, invalid_input_behavior.

Return exactly one valid JSON object. No markdown. No explanation.

---
ALLOWED VALUES
---

source:
  signature | type_hint | explicit | example | strongly_implied | inferred

precondition.kind  (never use "type"):
  domain | structural | relational | format | membership | numeric_range

postcondition.kind:
  semantic | relational | ordering | membership | numeric | structural

invariant.target:
  state | input | output | collection_element

invalid_input_behavior.source:
  explicit | example | strongly_implied | not_specified

---
HOW TO FILL EACH FIELD
---

task.summary
  One sentence. Say what the function computes.
  Right:  "Returns the count of integers in the list that are divisible by k."
  Wrong:  "Implement the function." / "This function does X."

interface.inputs
  One entry per parameter.
  description: explain the parameter's role in this task. Do not copy its type.
  Right:  "The exclusive lower bound; elements must strictly exceed this value."
  Wrong:  "Input parameter threshold."
  If the parameter has no type annotation, infer its type from the docstring
  and write both the inferred type and its role in the description.

interface.output
  description: explain what the return value represents. Do not write its type.
  Right:  "The number of elements in the list that exceed the threshold."
  Wrong:  "Return value produced by the function."

preconditions
  Write only constraints that the prompt, docstring, or examples explicitly state.
  Do not invent constraints. Do not add non-empty, positive, sorted, or unique
  requirements unless the prompt says so.
  Every target must be an exact parameter name from the signature.
  Never use kind "type" — type constraints are handled separately.

  Right:
    { "target": "n", "kind": "numeric_range",
      "description": "n must be greater than zero.",
      "source": "explicit" }

  Wrong — invented constraint:
    { "target": "lst", "kind": "structural",
      "description": "lst must not be empty.",
      "source": "strongly_implied" }

  Wrong — type constraint:
    { "target": "x", "kind": "type",
      "description": "x must be an integer.",
      "source": "signature" }

postconditions
  Write 1 to 3 postconditions that describe exactly what the function returns
  or what externally visible behavior it must produce.

  Use kind "semantic" for the main behavioural guarantee.
  Use additional postconditions only when the prompt clearly requires distinct
  observable behaviours, such as:
    - returning multiple values
    - returning a transformed DataFrame and a plot object
    - preserving exact DataFrame columns, index, shape, order, or dtype
    - setting exact plot title, labels, legend, colors, or bins
    - calling a specified API with a specified payload
    - writing, extracting, or returning a specific file or path
    - raising a specified exception with a specified message

  Do not invent postconditions.
  Do not include implementation details unless the prompt explicitly requires them.
  Every postcondition must be testable from the output or visible side effect.

edge_cases
  Include a case only when the prompt or its examples clearly support it.
  Write the specific input scenario in "case".
  Write the exact expected return value in "expected_behavior".
  Use [] when no case is clearly supported by the prompt.
  Do not include invalid inputs here.

  Right:
    { "case": "empty list input", "expected_behavior": "return 0",
      "source": "strongly_implied" }

  Wrong — placeholder text:
    { "case": "valid boundary case", "expected_behavior": "expected behavior",
      "source": "explicit" }

invariants
  Use [] for standalone functions. This is correct in almost all cases.
  Add an invariant only when the prompt describes a property that must hold
  throughout execution, not just at the start or end.

invalid_input_behavior
  Leave at the default (specified: false) unless the prompt explicitly says
  what happens for invalid input, such as "raise ValueError" or "return -1".
  Only then set specified to true and fill expected_behavior and exception_type.

---
EXAMPLE — study the style, then apply it to the actual function below
---

Do not copy these parameter names or values into your contract.
Read the actual function provided after these rules and fill values for that function.

Function:
  def count_above(numbers: List[int], threshold: int) -> int:
      \"\"\"Return the count of integers in numbers strictly greater than threshold.\"\"\"

Contract:

  task.summary:
    "Returns the count of integers in the input list that are strictly
     greater than the given threshold."

  interface.inputs:
    [ { "name": "numbers", "type": "List[int]",
        "description": "The list of integers to search through.",
        "source": "signature" },
      { "name": "threshold", "type": "int",
        "description": "The exclusive lower bound. Elements must be strictly
                        greater than this value to be counted.",
        "source": "signature" } ]

  interface.output:
    { "type": "int",
      "description": "The number of elements in numbers that are strictly
                      greater than threshold." }

  preconditions: []

  postconditions:
    [ { "id": "Q1", "target": "return", "kind": "semantic",
        "description": "result is the count of elements x in numbers such
                        that x > threshold, counting duplicates independently.",
        "source": "explicit" } ]

  edge_cases:
    [ { "id": "E1", "case": "numbers is an empty list",
        "expected_behavior": "return 0",
        "source": "strongly_implied" } ]

  invariants: []

  invalid_input_behavior: { "specified": false, "expected_behavior": "not_specified",
                            "exception_type": null, "description": "", "source": "not_specified" }

---
REQUIREMENTS — your JSON must satisfy all of these
---

  - task.summary states what is computed, not just the function name.
  - every input description explains the parameter's role in this specific task.
  - output description explains what the return value represents.
  - every precondition target is an exact parameter name from the signature.
  - no precondition uses kind "type".
  - every precondition constraint is explicitly stated or directly implied by the prompt.
  - the postcondition description is specific enough to describe only this function.
  - edge_cases contain only valid inputs, not invalid or error cases.
  - invalid_input_behavior.specified is false unless the prompt explicitly defines it.
  - each postcondition is specific enough to describe only this function.
- use multiple postconditions only for distinct observable requirements.
- do not invent plotting, file, API, exception, or invalid-input behavior.
""".strip()