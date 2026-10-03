from __future__ import annotations
import ast
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.generation.generation_core import normalize_prompt_text, taco_function_name, taco_question
JsonDict = dict[str, Any]
VERIFIER_VERSION = 'contract_quality2_verifier_20260928'
FINAL_STATUSES = frozenset({'CORRECT', 'DEFECTIVE', 'UNCERTAIN'})
CHECK_NAMES = ('interface', 'preconditions', 'postconditions', 'invariants', 'task_coverage', 'expression_description')
CHECK_VALUES = frozenset({'OK', 'ISSUE', 'UNCERTAIN'})
ISSUE_TYPES = frozenset({'incorrect_clause', 'vague_clause', 'missing_clause', 'expression_description_mismatch'})
ISSUE_ACTIONS = {'incorrect_clause': frozenset({'replace', 'remove'}), 'vague_clause': frozenset({'clarify', 'replace'}), 'missing_clause': frozenset({'add'}), 'expression_description_mismatch': frozenset({'align', 'replace'})}
TOP_KEYS = ('interface', 'preconditions', 'postconditions', 'invariants')
SECTIONS = ('preconditions', 'postconditions', 'invariants')
CLAUSE_KEYS = frozenset({'id', 'expression', 'description'})
INTERFACE_MODES = frozenset({'functional', 'stdin_stdout'})
LOCATION_RE = re.compile('^(preconditions|postconditions|invariants)\\[(\\d+)\\](?:\\.(expression|description))?$')
GENERIC_INSTRUCTION_RE = re.compile('^(fix|correct|repair|clarify|align|replace|add|remove)( this| the)?( clause| issue| requirement| expression| description)?[.!]?$', re.IGNORECASE)
FORBIDDEN_CALLS = frozenset({'input', 'print', 'exec', 'eval', 'open', '__import__', 'compile', 'setattr', 'delattr'})
MUTATING_METHODS = frozenset({'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'sort', 'reverse', 'update', 'setdefault', 'add', 'discard'})
BLOCKING_REPRESENTATION_CODES = frozenset({'invalid_interface', 'invalid_interface_mode', 'section_not_list', 'clause_not_object', 'invalid_expression', 'invalid_expression_syntax', 'assignment_expression_not_allowed', 'ellipsis_not_allowed', 'side_effect_or_dynamic_call_not_allowed', 'all_any_requires_iterable_expression', 'mutation_not_allowed_in_expression', 'non_boolean_result', 'invalid_all_any_argument'})

def large_model_validation_mode(model: Any) -> bool:
    model_type = str(getattr(model, 'model_type', '')).lower()
    identity = ' '.join((str(getattr(model, 'model_path', '')), str(getattr(model, 'adapter_path', '')))).lower()
    if model_type == 'qwen2' and ('32b' in identity or 'qwen32' in identity):
        return True
    if model_type == 'qwen3_5' and ('27b' in identity or 'qwen35' in identity or 'qwen3.5' in identity):
        return True
    if model_type == 'mistral3' and ('24b' in identity or 'devstral' in identity):
        return True
    return False

def blocking_representation_issues(analysis: JsonDict) -> list[JsonDict]:
    issues = analysis.get('representation_issues')
    if not isinstance(issues, list):
        return []
    blocking: list[JsonDict] = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        code = item.get('code')
        if code == 'missing_top_level_keys':
            detail = item.get('detail')
            missing = set(detail) if isinstance(detail, list) else set()
            if missing & {'interface', 'postconditions'}:
                blocking.append(item)
            continue
        if code in BLOCKING_REPRESENTATION_CODES:
            blocking.append(item)
    return blocking

class VerifierModel(Protocol):

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int=..., max_context: int=...) -> str:
        ...
SEMANTIC_AUDIT_SYSTEM_PROMPT = """
Verify CURRENT CONTRACT against ORIGINAL TASK. ORIGINAL TASK is the sole semantic source of truth. REQUIRED INTERFACE supplies only the calling convention.

Use an EXPRESSION-LED verification policy. Machine-readable expressions are the primary verification target. Natural-language descriptions are secondary and may affect a defect decision only when they define a semantic variable/predicate/helper needed to interpret an expression, or directly contradict an expression in a way that changes implementation-relevant meaning.

Do NOT mark a contract DEFECTIVE merely because a description is awkward, imprecise, incomplete in prose, out of context, redundant, or could be written better when the expression already captures the task correctly. Such prose-only concerns are at most UNCERTAIN and should normally be tolerated.

Audit exactly: interface, preconditions, postconditions, invariants, task_coverage, expression_description.

Expression-led rules:
- interface must match the required calling convention
- precondition EXPRESSIONS may contain only task-supported valid-input assumptions; output behavior is not a precondition
- postcondition EXPRESSIONS must capture the substantive required input-output relation
- preserve material branches, edge cases, quantifier scope, counting/decision semantics, constructive validity, impossibility, feasibility, optimization, global optimality, ordering, tie-breaking, modulo behavior, and per-case/query correspondence when applicable
- invariants are only genuine persistent/non-mutation properties; [] is valid
- missing_clause means a material task requirement is absent from the expression-level semantics of the whole contract
- vague_clause is reportable only when the expression itself is materially underspecified/opaque and its description does not define it precisely
- expression_description_mismatch is reportable only for a direct material contradiction; do not report harmless wording differences
- if the expression is task-correct and only the description is questionable, do NOT create a repair-triggering issue
- helper names are not oracles; use descriptions only when they explicitly define helpers needed to interpret expressions
- examples clarify explicit requirements but create no new requirements
- never invent bounds, uniqueness, ordering, sentinels, restrictions, state requirements, or algorithms
- if a defect is plausible but not confidently established, use UNCERTAIN

Allowed issue types/actions exactly:
incorrect_clause -> replace or remove
vague_clause -> clarify or replace
missing_clause -> add
expression_description_mismatch -> align or replace

Every repair-triggering issue must identify a substantive expression-level or interface defect. Do not emit an issue whose only target is prose quality.

Each issue must contain exactly:
{"type":"...","location":"...","action":"...","instruction":"...","task_evidence":"..."}

Evidence: task-grounded add/replace/clarify requires a short VERBATIM ORIGINAL TASK quote; unsupported whole-clause removal and expression_description_mismatch may use "". Never paraphrase task_evidence.

Locations: interface, interface.<field>, preconditions, postconditions, invariants, preconditions[index], postconditions[index], invariants[index], optionally followed by .expression or .description. Indexes are zero-based.

Important: a prose-only concern at *.description must NOT be returned as a repair issue. When an expression is wrong and its description must be repaired with it, target the whole clause or .expression; the repair stage may update the semantic pair together.

Return exactly:
{"checks":{"interface":"OK|ISSUE|UNCERTAIN","preconditions":"OK|ISSUE|UNCERTAIN","postconditions":"OK|ISSUE|UNCERTAIN","invariants":"OK|ISSUE|UNCERTAIN","task_coverage":"OK|ISSUE|UNCERTAIN","expression_description":"OK|ISSUE|UNCERTAIN"},"issues":[]}

Return JSON only.


Check meaning before proposing an edit:
- For one-based inclusive [L,R], Python items[L-1:R] has the correct endpoints.
  Derive the task's actual indexing convention; never add 1 automatically.
- A conditional must produce a Boolean on both branches: compare an output
  against a parenthesized conditional result, or compare in each branch.
- Check singleton and boundary ranges, distinct objects versus occurrences,
  quantifier dependence, necessary AND sufficient decision conditions, and
  chronological updates before queries. Do not guess a parity rule.
- all(... zip(...)) alone does not specify output cardinality. Look for a
  cardinality obligation elsewhere before reporting it missing.
- A helper that only names the desired answer is not a definition. Require
  an exact, non-circular relation, with arguments and relevant state defined.
- Check relevant examples printed in ORIGINAL TASK for contradictions. Use
  no external tests or solutions. Examples do not add unstated requirements.
- STATIC DIAGNOSTICS are review hints, not proof of semantic defects. Check
  their context before creating an issue. Do not execute contract expressions.
""".strip()
ISSUE_CONFIRMATION_SYSTEM_PROMPT = """
Independently confirm or reject the supplied candidate semantic defects. ORIGINAL TASK is the sole semantic source of truth. Do not search for another issue and do not repair the contract.

Use the same EXPRESSION-LED policy as the main verifier.

CONFIRMED requires that the interface is clearly wrong OR the machine-readable expression-level semantics are clearly wrong, unsupported, incomplete, or opaque relative to ORIGINAL TASK; CURRENT CONTRACT really contains/omits the claimed semantics; location/action are appropriate; and the instruction would make a real implementation-relevant correction.

Descriptions are secondary. Use them only to interpret explicit semantic definitions required by an expression or to identify a direct material expression-description contradiction.

Critical preservation rule: if the relevant expression already faithfully represents ORIGINAL TASK and the only problem is wording, context, completeness, or accuracy of the description, do NOT confirm the issue. Return REJECTED or UNCERTAIN.

For missing_clause inspect expression-level coverage across the whole contract. For vague_clause confirm only when the expression itself is materially vague and the description fails to define it. For expression_description_mismatch confirm only when the mismatch materially changes expression interpretation or the expression itself is wrong relative to the task. For remove confirm that the expression-level requirement is genuinely unsupported.

Reject style differences, redundancy, equivalent formulations, requirements already represented elsewhere, example-only inferences, invented assumptions, and unsupported restrictions. If plausible but not decisive, return UNCERTAIN.

Return exactly one:
{"decision":"CONFIRMED","issue_index":0}
{"decision":"REJECTED","issue_index":null}
{"decision":"UNCERTAIN","issue_index":null}

Do not rewrite the issue. Return JSON only.


A verbatim quote proves provenance only, not the alleged defect. Independently
derive the actual indexing, quantifier bounds, multiplicity, branches and
state semantics relevant to the issue. Check whether the old expression is
already equivalent to the task. Never confirm an unnecessary endpoint change.
Reject unsupported parity shortcuts and helpers that merely rename an answer.
Use relevant examples in ORIGINAL TASK to expose contradictions, not to infer
new constraints. If the supplied issues are rejected, that rejects those
allegations only; it does not certify every aspect of the contract.
""".strip()

def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _norm(value: Any) -> str:
    return ' '.join(str(value or '').casefold().split())

def _record(code: str, location: str | None=None, detail: Any=None) -> JsonDict:
    out: JsonDict = {'code': code}
    if location is not None:
        out['location'] = location
    if detail is not None:
        out['detail'] = detail
    return out

def _required_interface(task: JsonDict) -> JsonDict:
    name = taco_function_name(task)
    if name:
        return {'mode': 'functional', 'function_name': name}
    return {'mode': 'stdin_stdout', 'signature': 'stdin -> stdout', 'input_channel': 'stdin', 'output_channel': 'stdout'}

def _extract_json_object(raw: str) -> JsonDict | None:
    text = str(raw or '').strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    for block in re.findall('```(?:json)?\\s*(\\{.*?\\})\\s*```', text, flags=re.IGNORECASE | re.DOTALL):
        try:
            value = json.loads(block)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None

def _dedupe_issues(issues: list[JsonDict]) -> list[JsonDict]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[JsonDict] = []
    for issue in issues:
        sig = (str(issue.get('type') or ''), str(issue.get('location') or ''), str(issue.get('action') or ''), str(issue.get('instruction') or ''))
        if sig not in seen:
            seen.add(sig)
            result.append(issue)
    return result

def _non_boolean_result(node: ast.AST) -> bool:
    """Only definite literal-result errors; unknown helper types are not rejected."""
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, bool)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.GeneratorExp,
                         ast.ListComp, ast.SetComp, ast.DictComp, ast.Lambda)):
        return True
    if isinstance(node, ast.IfExp):
        return _non_boolean_result(node.body) or _non_boolean_result(node.orelse)
    if isinstance(node, ast.BoolOp):
        return any(_non_boolean_result(item) for item in node.values)
    return False


def _expression_issues(expression: Any, location: str, *, allow_iterable_expressions: bool=True) -> list[JsonDict]:
    if not isinstance(expression, str) or not expression.strip():
        return [_record('invalid_expression', location)]
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError):
        return [_record('invalid_expression_syntax', location)]
    issues = []
    if _non_boolean_result(tree.body):
        issues.append(_record('non_boolean_result', location,
                             'At least one result branch is a non-Boolean literal or container.'))
    for node in ast.walk(tree):
        if isinstance(node, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            issues.append(_record('assignment_expression_not_allowed', location))
        elif isinstance(node, ast.Constant) and node.value is Ellipsis:
            issues.append(_record('ellipsis_not_allowed', location))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in FORBIDDEN_CALLS:
                    issues.append(_record('side_effect_or_dynamic_call_not_allowed', location, name))
                if name in {'all', 'any'}:
                    if len(node.args) != 1 or node.keywords:
                        issues.append(_record('invalid_all_any_argument', location, name))
                    elif isinstance(node.args[0], ast.Constant) and not isinstance(node.args[0].value, (str, bytes)):
                        issues.append(_record('invalid_all_any_argument', location, name))
                    elif isinstance(node.args[0], ast.GeneratorExp):
                        gen = node.args[0]
                        for index, item in enumerate(gen.generators):
                            bound = {n.id for n in ast.walk(item.target) if isinstance(n, ast.Name)}
                            scope = [gen.elt, *item.ifs]
                            for later in gen.generators[index+1:]:
                                scope.extend([later.iter, *later.ifs])
                            used = {n.id for part in scope for n in ast.walk(part)
                                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                            if bound and not (bound & used):
                                issues.append(_record('unused_quantified_variable_hint', location, sorted(bound)))
                if name == 'zip':
                    issues.append(_record('zip_cardinality_review_hint', location,
                                         'Check output count elsewhere in the contract; zip stops at the shorter input.'))
            elif isinstance(node.func, ast.Attribute) and node.func.attr in MUTATING_METHODS:
                issues.append(_record('mutation_not_allowed_in_expression', location, node.func.attr))
    return issues

def _deterministic_interface_issue(task: JsonDict, contract: JsonDict) -> JsonDict | None:
    expected = _required_interface(task)
    interface = contract.get('interface')
    if not isinstance(interface, dict):
        return None
    expected_mode = expected['mode']
    actual_mode = str(interface.get('mode') or '').strip()
    if actual_mode in INTERFACE_MODES and actual_mode != expected_mode:
        return {'type': 'incorrect_clause', 'location': 'interface', 'action': 'replace', 'instruction': f'Replace the interface mode with {expected_mode!r} to match the required task calling convention.', 'task_evidence': '', 'source': 'deterministic'}
    if expected_mode == 'functional':
        expected_name = str(expected.get('function_name') or '')
        actual_name = str(interface.get('function_name') or '')
        if actual_mode == 'functional' and expected_name and (actual_name != expected_name):
            return {'type': 'incorrect_clause', 'location': 'interface.function_name', 'action': 'replace', 'instruction': f'Replace the function name with {expected_name!r} to match the required task interface.', 'task_evidence': '', 'source': 'deterministic'}
    if expected_mode == 'stdin_stdout' and actual_mode == 'stdin_stdout':
        signature = interface.get('signature')
        input_channel = interface.get('input_channel')
        output_channel = interface.get('output_channel')
        if signature is not None and signature != 'stdin -> stdout':
            return {'type': 'incorrect_clause', 'location': 'interface.signature', 'action': 'replace', 'instruction': 'Use the required stdin -> stdout interface.', 'task_evidence': '', 'source': 'deterministic'}
        if input_channel is not None and input_channel != 'stdin':
            return {'type': 'incorrect_clause', 'location': 'interface.input_channel', 'action': 'replace', 'instruction': 'Use stdin as the input channel.', 'task_evidence': '', 'source': 'deterministic'}
        if output_channel is not None and output_channel != 'stdout':
            return {'type': 'incorrect_clause', 'location': 'interface.output_channel', 'action': 'replace', 'instruction': 'Use stdout as the output channel.', 'task_evidence': '', 'source': 'deterministic'}
    return None

def analyze_contract(task: JsonDict, contract: Any, *, allow_iterable_expressions: bool=False) -> JsonDict:
    if not isinstance(contract, dict):
        return {'semantic_issues': [], 'representation_issues': [_record('contract_not_object', 'contract')]}
    semantic: list[JsonDict] = []
    representation: list[JsonDict] = []
    actual_keys = set(contract)
    required_keys = set(TOP_KEYS)
    missing = sorted(required_keys - actual_keys)
    extra = sorted(actual_keys - required_keys)
    if missing:
        representation.append(_record('missing_top_level_keys', 'contract', missing))
    if extra:
        representation.append(_record('extra_top_level_keys', 'contract', extra))
    interface = contract.get('interface')
    if not isinstance(interface, dict):
        representation.append(_record('invalid_interface', 'interface'))
    else:
        mode = interface.get('mode')
        if mode not in INTERFACE_MODES:
            representation.append(_record('invalid_interface_mode', 'interface.mode', repr(mode)))
        if mode == 'stdin_stdout':
            if 'signature' in interface and interface.get('signature') != 'stdin -> stdout':
                representation.append(_record('stdin_stdout_signature_mismatch', 'interface.signature'))
            if 'input_channel' in interface and interface.get('input_channel') != 'stdin':
                representation.append(_record('stdin_channel_mismatch', 'interface.input_channel'))
            if 'output_channel' in interface and interface.get('output_channel') != 'stdout':
                representation.append(_record('stdout_channel_mismatch', 'interface.output_channel'))
    interface_issue = _deterministic_interface_issue(task, contract)
    if interface_issue is not None:
        semantic.append(interface_issue)
    prefixes = {'preconditions': 'P', 'postconditions': 'Q', 'invariants': 'I'}
    for section in SECTIONS:
        clauses = contract.get(section)
        if not isinstance(clauses, list):
            representation.append(_record('section_not_list', section))
            continue
        if section == 'postconditions' and (not clauses):
            semantic.append({'type': 'missing_clause', 'location': 'postconditions', 'action': 'add', 'instruction': 'Add a substantive postcondition specifying the required input-output behavior from the original task.', 'task_evidence': '', 'source': 'deterministic'})
        for index, clause in enumerate(clauses):
            location = f'{section}[{index}]'
            if not isinstance(clause, dict):
                representation.append(_record('clause_not_object', location))
                continue
            if set(clause) != CLAUSE_KEYS:
                representation.append(_record('clause_keys_mismatch', location, {'expected': sorted(CLAUSE_KEYS), 'actual': sorted(clause)}))
            clause_id = clause.get('id')
            if not isinstance(clause_id, str) or not clause_id.strip():
                representation.append(_record('invalid_clause_id', f'{location}.id'))
            elif not clause_id.startswith(prefixes[section]):
                representation.append(_record('clause_id_prefix_mismatch', f'{location}.id', prefixes[section]))
            representation.extend(_expression_issues(clause.get('expression'), f'{location}.expression', allow_iterable_expressions=allow_iterable_expressions))
            description = clause.get('description')
            if not isinstance(description, str) or not description.strip():
                representation.append(_record('invalid_description', f'{location}.description'))
            if section == 'preconditions':
                text = _norm(str(clause.get('expression') or '') + ' ' + str(description or ''))
                if 'return_value' in text or re.search('\\bstdout\\b', text):
                    representation.append(_record('output_reference_in_precondition', location))
    return {'semantic_issues': _dedupe_issues(semantic), 'representation_issues': representation}

def _location_valid(location: str, contract: JsonDict) -> bool:
    if location == 'interface':
        return True
    if location.startswith('interface.'):
        return bool(location.split('.', 1)[1])
    if location in SECTIONS:
        return True
    match = LOCATION_RE.fullmatch(location)
    if match is None:
        return False
    section, raw_index, _ = match.groups()
    clauses = contract.get(section)
    return isinstance(clauses, list) and 0 <= int(raw_index) < len(clauses)

def _empty_evidence_allowed(issue: JsonDict) -> bool:
    return issue['type'] == 'expression_description_mismatch' or (issue['type'] == 'incorrect_clause' and issue['action'] == 'remove')

def _evidence_grounded(question: str, evidence: str) -> bool:
    evidence_norm = _norm(evidence)
    return bool(evidence_norm and evidence_norm in _norm(question))

def _normalize_issue(raw: Any, contract: JsonDict) -> tuple[JsonDict | None, str | None]:
    if not isinstance(raw, dict):
        return (None, 'issue is not an object')
    required = {'type', 'location', 'action', 'instruction', 'task_evidence'}
    if set(raw) != required:
        return (None, 'issue keys must be exactly ' + ','.join(sorted(required)))
    issue_type = str(raw.get('type') or '').strip()
    location = str(raw.get('location') or '').strip()
    action = str(raw.get('action') or '').strip()
    instruction = str(raw.get('instruction') or '').strip()
    evidence = str(raw.get('task_evidence') or '').strip()
    if issue_type not in ISSUE_TYPES:
        return (None, f'unsupported issue type {issue_type!r}')
    if action not in ISSUE_ACTIONS[issue_type]:
        return (None, f'invalid action {action!r} for {issue_type!r}')
    if not _location_valid(location, contract):
        return (None, f'invalid location {location!r}')
    if not instruction:
        return (None, 'empty instruction')
    if GENERIC_INSTRUCTION_RE.fullmatch(instruction):
        return (None, 'instruction is too generic')
    return ({'type': issue_type, 'location': location, 'action': action, 'instruction': instruction, 'task_evidence': evidence, 'source': 'semantic_audit'}, None)

def normalize_audit_response(raw: Any, *, task: JsonDict, contract: JsonDict) -> JsonDict:
    if not isinstance(raw, dict):
        return {'schema_valid': False, 'schema_error': 'audit is not an object'}
    if set(raw) != {'checks', 'issues'}:
        return {'schema_valid': False, 'schema_error': 'audit keys must be exactly checks,issues'}
    checks_raw = raw.get('checks')
    issues_raw = raw.get('issues')
    if not isinstance(checks_raw, dict) or set(checks_raw) != set(CHECK_NAMES):
        return {'schema_valid': False, 'schema_error': 'checks must contain exactly ' + ','.join(CHECK_NAMES)}
    checks: JsonDict = {}
    for name in CHECK_NAMES:
        value = str(checks_raw.get(name) or '').strip().upper()
        if value not in CHECK_VALUES:
            return {'schema_valid': False, 'schema_error': f'invalid {name} check {value!r}'}
        checks[name] = value
    if not isinstance(issues_raw, list):
        return {'schema_valid': False, 'schema_error': 'issues must be a list'}
    question = taco_question(task)
    issues: list[JsonDict] = []
    uncertainties: list[JsonDict] = []
    for index, raw_issue in enumerate(issues_raw):
        issue, error = _normalize_issue(raw_issue, contract)
        if issue is None:
            return {'schema_valid': False, 'schema_error': f'issue[{index}] schema invalid: {error}'}
        location = issue['location']
        if location.endswith('.description'):
            issue['location'] = location.rsplit('.', 1)[0]
            location = issue['location']
        evidence = issue['task_evidence']
        grounded = _evidence_grounded(question, evidence) if evidence else _empty_evidence_allowed(issue)
        issue['task_evidence_grounded'] = bool(grounded)
        if not grounded:
            uncertainties.append({'code': 'unverified_task_evidence' if evidence else 'missing_task_evidence', 'candidate_index': index, 'location': location})
        issues.append(issue)
    issue_checks = [name for name, value in checks.items() if value == 'ISSUE']
    if issue_checks and (not issues):
        uncertainties.append({'code': 'issue_check_without_substantive_candidate', 'checks': issue_checks})
    return {'schema_valid': True, 'checks': checks, 'issues': _dedupe_issues(issues), 'uncertainties': uncertainties}

def normalize_confirmation_response(raw: Any, *, candidate_count: int) -> JsonDict:
    if not isinstance(raw, dict):
        return {'schema_valid': False, 'schema_error': 'confirmation is not object'}
    if set(raw) != {'decision', 'issue_index'}:
        return {'schema_valid': False, 'schema_error': 'confirmation keys must be decision,issue_index'}
    decision = str(raw.get('decision') or '').strip().upper()
    index = raw.get('issue_index')
    if decision not in {'CONFIRMED', 'REJECTED', 'UNCERTAIN'}:
        return {'schema_valid': False, 'schema_error': f'invalid confirmation decision {decision!r}'}
    if decision == 'CONFIRMED':
        if not isinstance(index, int) or isinstance(index, bool) or (not 0 <= index < candidate_count):
            return {'schema_valid': False, 'schema_error': 'CONFIRMED needs valid integer issue_index'}
    elif index is not None:
        return {'schema_valid': False, 'schema_error': f'{decision} needs issue_index=null'}
    return {'schema_valid': True, 'decision': decision, 'issue_index': index}

def _call_json(model: VerifierModel, *, system_prompt: str, user_prompt: str, normalizer, max_new_tokens: int, max_context: int, json_retries: int, label: str) -> JsonDict:
    attempts = max(1, int(json_retries) + 1)
    last_raw = ''
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        prompt = user_prompt
        if attempt > 1:
            prompt = normalize_prompt_text(user_prompt + '\n\nSTRUCTURED-OUTPUT RETRY:\n' + errors[-1] + '\nCorrect only the JSON/schema problem. Return JSON only.')
        last_raw = model.generate(system_prompt, prompt, max_new_tokens=max_new_tokens, max_context=max_context)
        parsed = _extract_json_object(last_raw)
        if parsed is None:
            errors.append(f'{label}: response was not parseable JSON')
            continue
        normalized = normalizer(parsed)
        if not normalized.get('schema_valid'):
            errors.append(f"{label}: {normalized.get('schema_error') or 'schema invalid'}")
            continue
        return {'ok': True, 'value': normalized, 'raw_response': last_raw, 'calls': attempt, 'errors': errors}
    return {'ok': False, 'value': None, 'raw_response': last_raw, 'calls': attempts, 'errors': errors}

def _audit_prompt(task: JsonDict, contract: JsonDict, tid: str | None) -> str:
    diagnostics = analyze_contract(task, contract, allow_iterable_expressions=True)['representation_issues']
    return normalize_prompt_text('\n'.join((
        f"TASK ID: {tid or 'unspecified'}", '', 'ORIGINAL TASK:', taco_question(task),
        '', 'REQUIRED INTERFACE:', compact(_required_interface(task)),
        '', 'CURRENT CONTRACT:', compact(contract),
        '', 'STATIC DIAGNOSTICS (review in context):', compact(diagnostics),
        '', 'Audit the complete contract once. Use only ORIGINAL TASK and its printed examples; '
        'do not use generated code, external tests, execution results, or reference solutions.')))

def _confirmation_prompt(task: JsonDict, contract: JsonDict, issues: list[JsonDict], tid: str | None) -> str:
    candidates = [{'index': i, 'type': issue['type'], 'location': issue['location'], 'action': issue['action'], 'instruction': issue['instruction'], 'task_evidence': issue['task_evidence']} for i, issue in enumerate(issues)]
    return normalize_prompt_text('\n'.join((f"TASK ID: {tid or 'unspecified'}", '', 'ORIGINAL TASK:', taco_question(task), '', 'REQUIRED INTERFACE:', compact(_required_interface(task)), '', 'CURRENT CONTRACT:', compact(contract), '', 'CANDIDATE ISSUES:', compact(candidates))))

def _actionable_uncertain_issue(candidates: list[JsonDict]) -> JsonDict | None:
    """Return one evidence-backed semantic issue that is safe to *attempt* repairing.

    This does not promote UNCERTAIN to DEFECTIVE. It only authorizes an
    exploratory repair whose candidate must still re-verify as CORRECT before
    it can replace the original contract.
    """
    for item in candidates:
        if not isinstance(item, dict):
            continue
        location = str(item.get('location') or '')
        if location.endswith('.description'):
            continue
        if item.get('task_evidence_grounded') is True:
            issue = deepcopy(item)
            issue['source'] = 'actionable_uncertain_candidate'
            return issue
    return None

def _report(*, status: str, reason: str, checks: JsonDict | None, issue: JsonDict | None, candidates: list[JsonDict], deterministic: JsonDict, uncertainties: list[JsonDict], audit_raw: str, confirmation_raw: str, model_calls: int, errors: list[str]) -> JsonDict:
    if status not in FINAL_STATUSES:
        raise ValueError(f'invalid verifier status {status!r}')
    confirmed = [deepcopy(issue)] if issue is not None else []
    repair_issue = deepcopy(issue) if status == 'DEFECTIVE' and issue is not None else None
    return {'verifier_version': VERIFIER_VERSION, 'status': status, 'decision_reason': reason, 'workflow_action': 'REPAIR' if repair_issue is not None else 'KEEP', 'repair_recommended': repair_issue is not None, 'repair_basis': 'confirmed_defect' if repair_issue is not None else None, 'repair_issue': repair_issue, 'checks': deepcopy(checks) if checks is not None else None, 'issue': deepcopy(issue) if issue is not None else None, 'issues': confirmed, 'confirmed_issues': confirmed, 'candidate_issues': deepcopy(candidates), 'structural_issues': deepcopy(deterministic['semantic_issues']), 'semantic_structural_issues': deepcopy(deterministic['semantic_issues']), 'representation_issues': deepcopy(deterministic['representation_issues']), 'uncertainties': deepcopy(uncertainties), 'model_calls': int(model_calls), 'raw_response': audit_raw, 'confirmation_raw_response': confirmation_raw, 'errors': list(errors)}

def verify_contract(model: VerifierModel, task: JsonDict, contract: Any, *, tid: str | None=None, verify_tokens: int=768, confirm_tokens: int=256, max_context: int=32768, json_retries: int=1, **_: Any) -> JsonDict:
    deterministic = analyze_contract(task, contract, allow_iterable_expressions=bool(getattr(model, 'allow_iterable_expressions', False)))
    deterministic_semantic = deterministic['semantic_issues']
    if deterministic_semantic:
        return _report(status='DEFECTIVE', reason='deterministic_task_semantic_defect', checks=None, issue=deterministic_semantic[0], candidates=deterministic_semantic, deterministic=deterministic, uncertainties=[], audit_raw='', confirmation_raw='', model_calls=0, errors=[])
    if not isinstance(contract, dict):
        return _report(status='UNCERTAIN', reason='contract_representation_unresolved', checks=None, issue=None, candidates=[], deterministic=deterministic, uncertainties=[{'code': 'contract_not_semantically_auditable'}], audit_raw='', confirmation_raw='', model_calls=0, errors=[])
    audit_call = _call_json(model, system_prompt=SEMANTIC_AUDIT_SYSTEM_PROMPT, user_prompt=_audit_prompt(task, contract, tid), normalizer=lambda value: normalize_audit_response(value, task=task, contract=contract), max_new_tokens=verify_tokens, max_context=max_context, json_retries=json_retries, label='semantic_audit')
    calls = int(audit_call['calls'])
    if not audit_call['ok']:
        return _report(status='UNCERTAIN', reason='semantic_audit_structured_output_unresolved', checks=None, issue=None, candidates=[], deterministic=deterministic, uncertainties=[{'code': 'semantic_audit_failed', 'detail': audit_call['errors'][-1] if audit_call['errors'] else None}], audit_raw=audit_call['raw_response'], confirmation_raw='', model_calls=calls, errors=audit_call['errors'])
    audit = audit_call['value']
    checks = audit['checks']
    candidates = audit['issues']
    uncertainties = list(audit['uncertainties'])
    if candidates:
        confirmation_call = _call_json(model, system_prompt=ISSUE_CONFIRMATION_SYSTEM_PROMPT, user_prompt=_confirmation_prompt(task, contract, candidates, tid), normalizer=lambda value: normalize_confirmation_response(value, candidate_count=len(candidates)), max_new_tokens=confirm_tokens, max_context=max_context, json_retries=json_retries, label='issue_confirmation')
        calls += int(confirmation_call['calls'])
        if not confirmation_call['ok']:
            uncertainties.append({'code': 'issue_confirmation_failed', 'detail': confirmation_call['errors'][-1] if confirmation_call['errors'] else None})
            report = _report(status='UNCERTAIN', reason='candidate_defect_confirmation_unresolved', checks=checks, issue=None, candidates=candidates, deterministic=deterministic, uncertainties=uncertainties, audit_raw=audit_call['raw_response'], confirmation_raw=confirmation_call['raw_response'], model_calls=calls, errors=audit_call['errors'] + confirmation_call['errors'])
            repair_issue = _actionable_uncertain_issue(candidates)
            if repair_issue is not None:
                report['workflow_action'] = 'REPAIR'
                report['repair_recommended'] = True
                report['repair_basis'] = 'grounded_candidate_confirmation_unresolved'
                report['repair_issue'] = repair_issue
            return report
        confirmation = confirmation_call['value']
        decision = confirmation['decision']
        if decision == 'CONFIRMED':
            issue = deepcopy(candidates[confirmation['issue_index']])
            issue['source'] = 'independent_confirmation'
            if getattr(model, 'require_grounded_confirmation', True) and issue.get('task_evidence_grounded') is not True:
                return _report(status='UNCERTAIN', reason='confirmed_candidate_has_ungrounded_evidence', checks=checks, issue=None, candidates=candidates, deterministic=deterministic, uncertainties=uncertainties + [{'code': 'confirmed_evidence_not_grounded'}], audit_raw=audit_call['raw_response'], confirmation_raw=confirmation_call['raw_response'], model_calls=calls, errors=audit_call['errors'] + confirmation_call['errors'])
            return _report(status='DEFECTIVE', reason='independently_confirmed_semantic_defect', checks=checks, issue=issue, candidates=candidates, deterministic=deterministic, uncertainties=uncertainties, audit_raw=audit_call['raw_response'], confirmation_raw=confirmation_call['raw_response'], model_calls=calls, errors=audit_call['errors'] + confirmation_call['errors'])
        uncertainties.append({'code': 'semantic_audit_confirmation_disagreement' if decision == 'REJECTED' else 'semantic_candidate_confirmation_uncertain', 'candidate_count': len(candidates)})
        report = _report(status='UNCERTAIN', reason='semantic_candidate_not_confirmed', checks=checks, issue=None, candidates=candidates, deterministic=deterministic, uncertainties=uncertainties, audit_raw=audit_call['raw_response'], confirmation_raw=confirmation_call['raw_response'], model_calls=calls, errors=audit_call['errors'] + confirmation_call['errors'])
        report['confirmation_decision'] = decision
        if decision == 'UNCERTAIN':
            repair_issue = _actionable_uncertain_issue(candidates)
            if repair_issue is not None:
                report['workflow_action'] = 'REPAIR'
                report['repair_recommended'] = True
                report['repair_basis'] = 'grounded_candidate_confirmation_uncertain'
                report['repair_issue'] = repair_issue
        return report
    unresolved_checks = {name: value for name, value in checks.items() if value != 'OK'}
    if unresolved_checks:
        uncertainties.append({'code': 'semantic_checks_not_clean', 'checks': unresolved_checks})
    blocking_representation = blocking_representation_issues(deterministic)
    if blocking_representation:
        uncertainties.append({'code': 'representation_issues_block_correct_certification', 'count': len(blocking_representation), 'codes': sorted({str(item.get('code') or '') for item in blocking_representation})})
    if uncertainties:
        return _report(status='UNCERTAIN', reason='semantic_or_representation_uncertainty', checks=checks, issue=None, candidates=[], deterministic=deterministic, uncertainties=uncertainties, audit_raw=audit_call['raw_response'], confirmation_raw='', model_calls=calls, errors=audit_call['errors'])
    return _report(status='CORRECT', reason='all_semantic_checks_clean', checks=checks, issue=None, candidates=[], deterministic=deterministic, uncertainties=[], audit_raw=audit_call['raw_response'], confirmation_raw='', model_calls=calls, errors=audit_call['errors'])
__all__ = ['large_model_validation_mode', 'VERIFIER_VERSION', 'FINAL_STATUSES', 'BLOCKING_REPRESENTATION_CODES', 'analyze_contract', 'blocking_representation_issues', 'verify_contract']
