CODE_GUIDANCE_RULES = """
Solve the programming task using the ORIGINAL TASK, REQUIRED INTERFACE, and CONTRACT.

Authority:
1. ORIGINAL TASK defines the required behavior.
2. REQUIRED INTERFACE defines how the solution must be exposed.
3. CONTRACT provides structured semantic guidance.

Use every contract clause that is consistent with the ORIGINAL TASK. If the contract omits a task requirement, implement that requirement from the ORIGINAL TASK. If a contract clause conflicts with the ORIGINAL TASK or introduces an unsupported requirement, ignore only that conflicting part and continue using the remaining task-consistent contract.

Interpret the contract semantically:
- Preconditions describe assumptions guaranteed for valid inputs. Do not turn them into assertions, exceptions, input rejection, or runtime validation unless the ORIGINAL TASK explicitly requires that behavior.
- Postconditions describe the required input-output relation. Use them to determine what the program must compute.
- Invariants describe persistent-state or non-mutation requirements when applicable.
- Interpret each expression together with its description.
- Semantic predicates, helper names, and specification variables describe required behavior; they are not executable functions or implementation oracles.
- Do not infer additional behavior from helper names alone.

Determine the complete task semantics before choosing the implementation. Preserve every applicable:
- branch and edge case
- quantifier scope
- necessary-and-sufficient decision condition
- counting rule, including ordering, multiplicity, and distinctness
- constructive validity condition
- impossibility condition
- feasibility condition
- optimization objective
- global optimality requirement
- secondary objective and tie-breaking rule
- ordering requirement
- modulo requirement
- testcase/query-to-output correspondence
- state-preservation requirement

For decision problems, implement the exact condition under which the required answer is true or false.

For counting problems, determine exactly which objects are counted and whether order, multiplicity, equality, and distinctness change the count.

For constructive problems, produce a witness satisfying every required condition. Output impossibility only when the ORIGINAL TASK's impossibility condition holds.

For optimization problems, first satisfy feasibility, then optimize exactly the required objective, including any secondary objective or tie-breaking rule.

Use the input constraints to choose the algorithm and data structures. The implementation must have sufficient time and memory complexity for the stated limits. Do not use exhaustive search or brute force when those limits require a more efficient method.

Preserve the REQUIRED INTERFACE exactly:
- For call-based tasks, define the required class/function/method with the required name and calling convention.
- For stdin/stdout tasks, parse the specified input format and print exactly the required output.
- Do not change return behavior, output shape, or per-testcase/query correspondence.

Do not:
- mechanically translate contract expressions into Python
- execute or evaluate contract expressions at runtime
- create contract-checking code
- invent helper semantics not defined by the task or contract
- add assertions, exceptions, validation, or rejection behavior not required by the ORIGINAL TASK
- add extra output
- replace a required efficient algorithm with a brute-force implementation

The final program must simultaneously satisfy:
- the ORIGINAL TASK behavior
- the REQUIRED INTERFACE
- all task-consistent contract obligations
- all input/output formatting requirements
- all relevant edge cases
- the required computational constraints

Return exactly one complete Python solution.
Return Python code only.
Do not include tests, reasoning, explanations, Markdown, or code fences.
""".strip()