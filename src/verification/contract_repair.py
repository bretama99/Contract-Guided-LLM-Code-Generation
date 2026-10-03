from __future__ import annotations
import ast
import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Protocol
from src.generation.generation_core import normalize_prompt_text, taco_function_name, taco_question
from src.verification.contract_verifier import analyze_contract, blocking_representation_issues, verify_contract, _call_json
JsonDict = dict[str, Any]
REPAIR_VERSION = 'contract_quality2_repair_20260928'
CLAUSE_RE = re.compile('^(preconditions|postconditions|invariants)\\[(\\d+)\\](?:\\.(expression|description))?$')
REPAIR_SYSTEM_PROMPT = """
Repair ONE authorized contract issue using ORIGINAL TASK as the sole semantic
source of truth. The issue may be a confirmed defect or an evidence-backed
UNCERTAIN candidate. Re-derive the target from ORIGINAL TASK rather than assuming
the verifier's candidate wording is correct.

Repair the complete semantic pair for the authorized target:
{"id":"...","expression":"...","description":"..."}

The EXPRESSION is primary. Re-derive it from ORIGINAL TASK before writing it.
Then write a description that states exactly the same semantics. Do not preserve
a wrong expression merely because its old description agrees with it.

Rules:
- change only the authorized interface/clause/section target
- preserve all unrelated contract content exactly
- make the smallest task-grounded semantic correction that fully removes the
  confirmed defect
- never invent bounds, assumptions, branches, sentinels, ordering, uniqueness,
  optimization rules, or implementation details
- preconditions contain only task-supported valid-input assumptions
- postconditions specify substantive input-output behavior
- invariants are only genuine persistent/non-mutation properties
- preserve branches, edge cases, quantifier scope, counting/decision semantics,
  constructive validity, impossibility, feasibility, optimization, ordering,
  tie-breaking, modulo behavior, and per-case/query correspondence
- every expression must be one Python boolean expression parseable by
  ast.parse(expression, mode="eval")
- no input(), print(), imports, mutation, assignments, comments, ellipsis, or
  natural-language fragments inside expressions
- stdin/stdout tasks use stdin and stdout as the full text channels
- functional postconditions use return_value for the returned value
- semantic helpers must be explicitly and non-circularly defined
- do not generate code, tests, assertions, or runtime contract checkers
- do not repeat a previously rejected candidate
- when verifier feedback from a previous candidate is supplied, use it to avoid
  the exact semantic weakness that prevented CORRECT certification

Return exactly:
{"value": ...}

Replacement/repair of a clause -> value is the COMPLETE
{"id","expression","description"} clause.
Adding a missing clause -> value is one COMPLETE clause.
SECTION fallback repair -> value is the COMPLETE array of clauses for that
single authorized section. Reconstruct that section from ORIGINAL TASK while
preserving every other contract section exactly.
Interface repair -> value is the COMPLETE interface object.

Return JSON only.


Before returning the value, silently check the actual changed expression:
- Derive index conventions. A one-based inclusive L..R slice is [L-1:R].
- Check both conditional branches are Boolean; prefer output == (a if c else b).
- Check the first/last element and singleton quantifier ranges; each bound
  variable must play its intended role in the relation.
- Distinguish distinct objects from occurrences and first from last occurrence.
- Derive decision rules from legal transitions; never guess parity shortcuts.
- Preserve output cardinality and case/query correspondence. zip alone cannot
  ensure every query receives an answer.
- For stateful tasks define update order, initial state and query-time state.
- Define every introduced semantic helper exactly and non-circularly. A name
  such as correct_answer or path_fee_sum without a definition is insufficient.
- Check relevant examples printed in ORIGINAL TASK; do not execute expressions
  or use external tests/solutions. Do not infer new restrictions from examples.
- On a section rewrite, preserve every previously correct obligation, including
  output counts, branches, ordering and tie-breaking. Remove an obligation only
  when the task shows it is unsupported. Preconditions and invariants may be [];
  postconditions must retain a substantive input-output relation.
""".strip()

class RepairModel(Protocol):

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int=..., max_context: int=...) -> str:
        ...

def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _fingerprint(contract: JsonDict) -> str:
    return hashlib.sha256(_dump(contract).encode('utf-8')).hexdigest()

def _extract_json(raw: str) -> JsonDict | None:
    text = str(raw or '').strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
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

def _valid_expression(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        tree = ast.parse(value, mode='eval')
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)):
            return False
        if isinstance(node, ast.Constant) and node.value is Ellipsis:
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {'input', 'print', 'exec', 'eval', 'open', '__import__'}:
                return False
    return True

def _valid_clause(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {'id', 'expression', 'description'} and isinstance(value.get('id'), str) and bool(value['id'].strip()) and _valid_expression(value.get('expression')) and isinstance(value.get('description'), str) and bool(value['description'].strip())

def _required_interface(task: JsonDict, current: Any) -> JsonDict:
    """Deterministically correct only objective interface facts."""
    name = taco_function_name(task)
    if not name:
        return {'mode': 'stdin_stdout', 'signature': 'stdin -> stdout', 'input_channel': 'stdin', 'output_channel': 'stdout'}
    interface = deepcopy(current) if isinstance(current, dict) else {}
    interface['mode'] = 'functional'
    interface['function_name'] = name
    interface.setdefault('class_name', None)
    interface.pop('input_channel', None)
    interface.pop('output_channel', None)
    if interface.get('signature') == 'stdin -> stdout':
        interface.pop('signature', None)
    return interface

def _semantic_issue(verification: JsonDict) -> JsonDict | None:
    issue = verification.get('issue')
    if isinstance(issue, dict):
        return issue
    issues = verification.get('confirmed_issues')
    if isinstance(issues, list):
        for item in issues:
            if isinstance(item, dict):
                return item
    return None

def _repairable_representation_issue(verification: JsonDict) -> JsonDict | None:
    """Return only objective BLOCKING representation defects.

    Harmless legacy-schema differences must never consume repair attempts.
    """
    blocking = blocking_representation_issues({'representation_issues': verification.get('representation_issues') or []})
    for item in blocking:
        location = str(item.get('location') or '')
        if location == 'interface' or location.startswith('interface.'):
            return {'type': 'representation', 'location': location, 'action': 'replace', 'instruction': str(item.get('code') or 'repair interface representation'), 'task_evidence': '', 'source': 'representation'}
        match = CLAUSE_RE.fullmatch(location)
        if match:
            return {'type': 'representation', 'location': location, 'action': 'replace', 'instruction': str(item.get('code') or 'repair clause representation'), 'task_evidence': '', 'source': 'representation'}
    return None

def _issue_section(issue: JsonDict | None) -> str | None:
    if not isinstance(issue, dict):
        return None
    location = str(issue.get('location') or '')
    if location in {'preconditions', 'postconditions', 'invariants'}:
        return location
    match = CLAUSE_RE.fullmatch(location)
    return match.group(1) if match else None

def _section_fallback_target(issue: JsonDict, attempt: int, *, early: bool=False) -> JsonDict:
    section = _issue_section(issue)
    threshold = 2 if early else 3
    if attempt < threshold or section is None:
        return issue
    if str(issue.get('location') or '').startswith('interface'):
        return issue
    if issue.get('action') == 'remove':
        return issue
    return {'type': issue.get('type') or 'incorrect_clause', 'location': section, 'action': 'replace', 'instruction': f'Targeted repair did not reach CORRECT. Reconstruct the complete {section} section from ORIGINAL TASK. Preserve correct semantics and fix all implementation-relevant semantic errors in this section.', 'task_evidence': issue.get('task_evidence') or '', 'source': 'section_fallback_after_targeted_failure', 'original_issue': deepcopy(issue)}

def choose_repair_target(verification: JsonDict) -> JsonDict | None:
    if verification.get('status') == 'DEFECTIVE':
        return _semantic_issue(verification)
    if verification.get('status') == 'UNCERTAIN' and verification.get('repair_recommended') is True:
        issue = verification.get('repair_issue')
        if isinstance(issue, dict):
            return deepcopy(issue)
    return _repairable_representation_issue(verification)

def _target_view(contract: JsonDict, issue: JsonDict) -> tuple[str, Any]:
    location = str(issue.get('location') or '')
    if location == 'interface' or location.startswith('interface.'):
        return ('interface', contract.get('interface'))
    match = CLAUSE_RE.fullmatch(location)
    if match:
        section, raw_index, _ = match.groups()
        index = int(raw_index)
        clauses = contract.get(section)
        if not isinstance(clauses, list) or not 0 <= index < len(clauses):
            raise ValueError(f'Invalid repair location: {location}')
        return ('clause', clauses[index])
    if location in {'preconditions', 'postconditions', 'invariants'}:
        if issue.get('action') == 'replace':
            current = contract.get(location)
            if not isinstance(current, list):
                raise ValueError(f'{location} is not a list')
            return ('section', current)
        if issue.get('action') == 'add' or issue.get('type') == 'missing_clause':
            return ('add_clause', None)
    raise ValueError(f'Unsupported repair location: {location!r}')

def _strategy(attempt: int) -> str:
    strategies = ('Minimal correction: fix only the confirmed semantic error.', 'Fresh derivation: ignore the old expression first and derive the target from ORIGINAL TASK.', 'Boundary audit: explicitly account for branches, edge cases, bounds, and quantifier scope.', 'Relation audit: reconstruct the exact required input-output/counting/decision relation.', 'Independent reconstruction: rebuild the authorized target from scratch, then check it against every relevant task requirement.')
    return strategies[(max(1, attempt) - 1) % len(strategies)]

def _repair_prompt(task: JsonDict, contract: JsonDict, issue: JsonDict, *, tid: str, attempt: int, previous_attempts: list[JsonDict]) -> str:
    kind, current_value = _target_view(contract, issue)
    prior = [{'attempt': item.get('attempt'), 'result': item.get('verification_status') or item.get('generation_status'), 'candidate_value': item.get('candidate_value'), 'reason': item.get('decision_reason'), 'checks': item.get('checks'), 'confirmed_issues': item.get('confirmed_issues'), 'candidate_issues': item.get('candidate_issues'), 'uncertainties': item.get('uncertainties'), 'generation_errors': item.get('generation_errors'), 'change_review': item.get('change_review')} for item in previous_attempts[-4:]]
    return normalize_prompt_text('\n'.join((f'TASK ID: {tid}', f'REPAIR ATTEMPT: {attempt}', f'TARGET KIND: {kind}', '', 'ORIGINAL TASK:', taco_question(task), '', 'CURRENT CONTRACT:', _dump(contract), '', 'REPAIR TARGET / EVIDENCE:', _dump(issue), '', 'CURRENT TARGET VALUE:', _dump(current_value), '', 'PREVIOUS ATTEMPTS:', _dump(prior), '', 'ATTEMPT STRATEGY:', _strategy(attempt), '', 'Silently derive the correct target semantics from ORIGINAL TASK first. For a clause target, return one complete repaired semantic pair. For a section target, reconstruct that ONE section completely enough to express the task faithfully; do not preserve a bad clause merely to keep the old section shape. Preserve everything outside the authorized target. Do not repeat any candidate shown in PREVIOUS ATTEMPTS.')))

def _normalize_value(kind: str, value: Any, current_value: Any, issue: JsonDict) -> Any:
    if kind == 'interface':
        return value if isinstance(value, dict) else None
    if kind == 'section':
        if not isinstance(value, list) or (not value and issue.get('location') == 'postconditions'):
            return None
        if len(value) > 12 or not all((_valid_clause(item) for item in value)):
            return None
        section = str(issue.get('location') or '')
        prefix = {'preconditions': 'P', 'postconditions': 'Q', 'invariants': 'I'}.get(section)
        if prefix is None:
            return None
        normalized = []
        for index, item in enumerate(value, start=1):
            clause = deepcopy(item)
            clause['id'] = f'{prefix}{index}'
            normalized.append(clause)
        return normalized
    if kind in {'clause', 'add_clause'}:
        if not _valid_clause(value):
            return None
        clause = deepcopy(value)
        if kind == 'clause' and isinstance(current_value, dict):
            current_id = current_value.get('id')
            if isinstance(current_id, str) and current_id.strip():
                clause['id'] = current_id
        return clause
    return None

def _apply_value(contract: JsonDict, issue: JsonDict, value: Any) -> JsonDict:
    candidate = deepcopy(contract)
    location = str(issue.get('location') or '')
    action = str(issue.get('action') or '')
    if location == 'interface' or location.startswith('interface.'):
        if not isinstance(value, dict):
            raise ValueError('Interface repair must be an object')
        candidate['interface'] = deepcopy(value)
        return candidate
    match = CLAUSE_RE.fullmatch(location)
    if match:
        section, raw_index, _ = match.groups()
        index = int(raw_index)
        clauses = candidate.get(section)
        if not isinstance(clauses, list) or not 0 <= index < len(clauses):
            raise ValueError(f'Invalid clause location: {location}')
        if action == 'remove':
            del clauses[index]
        else:
            if not _valid_clause(value):
                raise ValueError('Clause repair must be a complete valid clause')
            clauses[index] = deepcopy(value)
        return candidate
    if location in {'preconditions', 'postconditions', 'invariants'}:
        if action == 'replace':
            if not isinstance(value, list) or (not value and location == 'postconditions') or (not all((_valid_clause(item) for item in value))):
                raise ValueError('Section repair must contain valid clauses; only postconditions must be non-empty')
            candidate[location] = deepcopy(value)
            return candidate
        if action != 'add':
            raise ValueError(f'Unsupported section action: {action}')
        if not _valid_clause(value):
            raise ValueError('Added clause must be a complete valid clause')
        prefix = {'preconditions': 'P', 'postconditions': 'Q', 'invariants': 'I'}[location]
        clause = deepcopy(value)
        clauses = candidate.get(location)
        if not isinstance(clauses, list):
            raise ValueError(f'{location} is not a list')
        used = {str(item.get('id')) for item in clauses if isinstance(item, dict)}
        number = 1
        while f'{prefix}{number}' in used:
            number += 1
        clause['id'] = f'{prefix}{number}'
        clauses.append(clause)
        return candidate
    raise ValueError(f'Unsupported repair location: {location}')

def _generate_candidate(model: RepairModel, task: JsonDict, contract: JsonDict, issue: JsonDict, *, tid: str, attempt: int, previous_attempts: list[JsonDict], repair_tokens: int, max_context: int, json_retries: int) -> JsonDict:
    """Generate only the authorized value; Python applies the patch locally."""
    kind, current_value = _target_view(contract, issue)
    if kind == 'interface':
        required = _required_interface(task, current_value)
        field = str(issue.get('location') or '').partition('.')[2]
        objective = field in {'mode', 'function_name', 'input_channel', 'output_channel'}
        objective = objective or (not taco_function_name(task) and field in {'', 'signature'})
        if not field and (not isinstance(current_value, dict) or current_value.get('mode') != required.get('mode') or current_value.get('function_name') != required.get('function_name')):
            objective = True
        if objective and required != current_value:
            candidate = _apply_value(contract, issue, required)
            return {'status': 'OK', 'candidate_contract': candidate, 'value': candidate['interface'], 'model_calls': 0, 'raw_response': '', 'errors': []}
    if issue.get('action') == 'remove':
        return {'status': 'OK', 'candidate_contract': _apply_value(contract, issue, None), 'value': None, 'model_calls': 0, 'raw_response': '', 'errors': []}
    base_prompt = _repair_prompt(task, contract, issue, tid=tid, attempt=attempt, previous_attempts=previous_attempts)
    errors: list[str] = []
    raw = ''
    calls = 0
    rejected_values = {_dump(item.get('candidate_value')) for item in previous_attempts if item.get('candidate_value') is not None}
    for retry in range(max(1, int(json_retries) + 1)):
        prompt = base_prompt
        if retry:
            prompt += '\n\nSTRUCTURED-OUTPUT RETRY:\n' + errors[-1] + '\nReturn exactly {"value":...}. For a section target, value must be the complete clause array for that section; otherwise return the required complete target.'
        raw = model.generate(REPAIR_SYSTEM_PROMPT, prompt, max_new_tokens=repair_tokens, max_context=max_context)
        calls += 1
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict) or set(parsed) != {'value'}:
            errors.append("response must contain exactly the key 'value'")
            continue
        value = _normalize_value(kind, parsed.get('value'), current_value, issue)
        if value is None:
            errors.append('returned target value has invalid structure or expression syntax')
            continue
        if kind == 'interface' and isinstance(current_value, dict):
            field = str(issue.get('location') or '').partition('.')[2]
            if field:
                if field not in value:
                    errors.append('returned interface omits the authorized field: ' + field)
                    continue
                replacement = deepcopy(current_value)
                replacement[field] = deepcopy(value[field])
                value = replacement
        if _dump(value) in rejected_values:
            errors.append('returned target exactly repeats a previously rejected candidate; derive a materially different task-grounded correction')
            continue
        candidate = _apply_value(contract, issue, value)
        analysis = analyze_contract(task, candidate, allow_iterable_expressions=bool(getattr(model, 'allow_iterable_expressions', False)))
        blocking = blocking_representation_issues(analysis)
        if blocking:
            errors.append('candidate failed blocking representation gate: ' + _dump(blocking[:5]))
            continue
        return {'status': 'OK', 'candidate_contract': candidate, 'value': value, 'model_calls': calls, 'raw_response': raw, 'errors': errors}
    return {'status': 'INVALID', 'candidate_contract': None, 'value': None, 'model_calls': calls, 'raw_response': raw, 'errors': errors}


CHANGE_REVIEW_SYSTEM_PROMPT = """
Review a proposed contract change against ORIGINAL TASK. Do not assume the
verifier allegation or the replacement is correct. Assess the change, not style.

Check each item:
change_supported: the task supports changing the original semantics; a literal
task quote alone is not proof. Equivalent restatements are not improvements.
target_correct: the replacement resolves the target without an index, branch,
quantifier, multiplicity, decision-rule, undefined-helper or state-order error.
obligations_preserved: every other task-supported obligation remains represented
in the complete candidate, especially output counts, ordering and failure cases.
examples_consistent: the changed relation contradicts no relevant printed task
example or task-derived boundary case. If no examples are relevant, use OK.

For one-based inclusive L..R, items[L-1:R] is correct. zip alone does not require
equal lengths. Check singleton ranges. Both conditional result branches must
be Boolean. Do not confuse occurrences with distinct values. A helper must have
an exact non-circular definition; naming the answer is not a specification.

Use only ORIGINAL TASK, the original contract, and the candidate. Do not use
external tests, reference solutions, code generation or execution. Use UNCERTAIN
when you cannot establish a check; use ISSUE for a concrete contradiction.
Return exactly one JSON object:
{"checks":{"change_supported":"OK|ISSUE|UNCERTAIN","target_correct":"OK|ISSUE|UNCERTAIN","obligations_preserved":"OK|ISSUE|UNCERTAIN","examples_consistent":"OK|ISSUE|UNCERTAIN"},"explanation":"One concise reason; include a concrete contradiction if found."}
""".strip()


def _normalize_change_review(value: Any) -> JsonDict:
    names = {'change_supported', 'target_correct', 'obligations_preserved', 'examples_consistent'}
    if not isinstance(value, dict) or set(value) != {'checks', 'explanation'}:
        return {'schema_valid': False, 'schema_error': 'Expected checks and explanation.'}
    checks = value['checks']
    if not isinstance(checks, dict) or set(checks) != names or any(not isinstance(v, str) or v not in {'OK', 'ISSUE', 'UNCERTAIN'} for v in checks.values()):
        return {'schema_valid': False, 'schema_error': 'Invalid change-review checks.'}
    if not isinstance(value['explanation'], str) or not value['explanation'].strip():
        return {'schema_valid': False, 'schema_error': 'A concise explanation is required.'}
    return {'schema_valid': True, 'checks': checks, 'explanation': value['explanation']}


def review_contract_change(model: RepairModel, task: JsonDict, original: JsonDict,
                           candidate: JsonDict, targets: list, args: Any) -> JsonDict:
    prompt = normalize_prompt_text('\n'.join((
        'ORIGINAL TASK:', taco_question(task), 'ORIGINAL CONTRACT:', _dump(original),
        'PROPOSED COMPLETE CONTRACT:', _dump(candidate),
        'REPAIR TARGET HISTORY (allegations, not facts):', _dump(targets),
        'Check the semantic difference and preserve all supported obligations.')))
    result = _call_json(model, system_prompt=CHANGE_REVIEW_SYSTEM_PROMPT,
                        user_prompt=prompt, normalizer=_normalize_change_review,
                        max_new_tokens=int(getattr(args, 'change_review_tokens', 768)),
                        max_context=int(getattr(args, 'max_context', 32768)),
                        json_retries=int(getattr(args, 'json_retries', 1)), label='change_review')
    value = result.get('value') or {}
    return {'accepted': bool(result['ok'] and all(x == 'OK' for x in value.get('checks', {}).values())),
            'checks': value.get('checks'), 'explanation': value.get('explanation'),
            'model_calls': result['calls'], 'errors': result['errors'],
            'raw_response': result['raw_response']}


def _target_key(target: JsonDict) -> str:
    # Rewording feedback about the same location must not restart the retry count.
    return str(target.get('location') or '').removesuffix('.expression').removesuffix('.description') + ':' + str(target.get('action') or '')


def repair_defective_contract(model: RepairModel, task: JsonDict, original_contract: JsonDict, original_verification: JsonDict, args: Any, *, tid: str) -> JsonDict:
    original_status = original_verification.get('status')
    if original_status == 'DEFECTIVE':
        pass
    elif original_status == 'UNCERTAIN' and original_verification.get('repair_recommended') is True and (choose_repair_target(original_verification) is not None):
        pass
    else:
        raise ValueError('repair requires DEFECTIVE or actionable UNCERTAIN verification')
    retries = max(0, int(getattr(args, 'retries', 0)))
    repair_tokens = int(getattr(args, 'repair_tokens', 768))
    verify_tokens = int(getattr(args, 'verify_tokens', 768))
    confirm_tokens = int(getattr(args, 'confirm_tokens', 256))
    max_context = int(getattr(args, 'max_context', 32768))
    json_retries = int(getattr(args, 'json_retries', 1))
    working_contract = deepcopy(original_contract)
    working_verification = deepcopy(original_verification)
    last_repairable_contract = deepcopy(original_contract)
    last_repairable_verification = deepcopy(original_verification)
    history: list[JsonDict] = []
    seen = {_fingerprint(original_contract)}
    target_attempts: dict[str, int] = {}
    for attempt in range(1, retries + 1):
        target = choose_repair_target(working_verification)
        if target is None and working_verification.get('status') == 'UNCERTAIN':
            working_contract = deepcopy(last_repairable_contract)
            working_verification = deepcopy(last_repairable_verification)
            target = choose_repair_target(working_verification)
        if target is None:
            history.append({'attempt': attempt, 'repair_kind': None, 'target': None, 'generation_status': 'NOT_RUN', 'candidate_value': None, 'candidate_fingerprint': None, 'verification_status': working_verification.get('status'), 'decision_reason': 'no_repairable_confirmed_issue', 'repair_model_calls': 0, 'verification_model_calls': 0})
            break
        key = _target_key(target)
        target_attempts[key] = target_attempts.get(key, 0) + 1
        target = _section_fallback_target(target, target_attempts[key], early=False)
        generated = _generate_candidate(model, task, working_contract, target, tid=tid, attempt=attempt, previous_attempts=history, repair_tokens=repair_tokens, max_context=max_context, json_retries=json_retries)
        candidate = generated.get('candidate_contract')
        base_event = {'attempt': attempt, 'repair_kind': 'interface' if str(target.get('location') or '').startswith('interface') else 'section_fallback' if target.get('source') == 'section_fallback_after_targeted_failure' else 'expression_semantic', 'target': deepcopy(target), 'generation_status': generated.get('status'), 'candidate_value': deepcopy(generated.get('value')), 'repair_model_calls': int(generated.get('model_calls') or 0), 'generation_errors': list(generated.get('errors') or [])}
        if not isinstance(candidate, dict):
            history.append({**base_event, 'candidate_fingerprint': None, 'verification_status': None, 'decision_reason': 'no_valid_repair_candidate', 'verification_model_calls': 0})
            continue
        candidate_hash = _fingerprint(candidate)
        if candidate_hash in seen:
            history.append({**base_event, 'candidate_fingerprint': candidate_hash, 'verification_status': None, 'decision_reason': 'duplicate_repair_candidate', 'verification_model_calls': 0})
            continue
        seen.add(candidate_hash)
        verification = verify_contract(model, task, candidate, tid=f'{tid}.repair{attempt}', verify_tokens=verify_tokens, confirm_tokens=confirm_tokens, max_context=max_context, json_retries=json_retries)
        status = verification.get('status')
        history.append({**base_event, 'candidate_fingerprint': candidate_hash, 'verification_status': status, 'decision_reason': verification.get('decision_reason'), 'verification_model_calls': int(verification.get('model_calls') or 0), 'checks': deepcopy(verification.get('checks')), 'confirmed_issues': deepcopy(verification.get('confirmed_issues') or []), 'candidate_issues': deepcopy((verification.get('candidate_issues') or [])[:5]), 'representation_issues': deepcopy((verification.get('representation_issues') or [])[:5]), 'uncertainties': deepcopy((verification.get('uncertainties') or [])[:5]), 'verification': deepcopy(verification)})
        if status == 'CORRECT':
            review = review_contract_change(model, task, original_contract, candidate,
                                            [x['target'] for x in history if x.get('target')], args)
            history[-1]['change_review'] = deepcopy(review)
            history[-1]['change_review_model_calls'] = int(review['model_calls'])
            if not review['accepted']:
                history[-1]['decision_reason'] = 'change_review_not_accepted'
                working_contract = deepcopy(last_repairable_contract)
                working_verification = deepcopy(last_repairable_verification)
                continue
            verification['change_review'] = deepcopy(review)
            return {'resolved': True, 'accepted_contract': candidate, 'accepted_verification': verification, 'attempts': history, 'attempts_used': len(history)}
        if status == 'DEFECTIVE' or (status == 'UNCERTAIN' and verification.get('repair_recommended') is True and (choose_repair_target(verification) is not None)):
            working_contract = candidate
            working_verification = verification
            last_repairable_contract = deepcopy(candidate)
            last_repairable_verification = deepcopy(verification)
            continue
        deterministic_patch = int(generated.get('model_calls') or 0) == 0 and (str(target.get('location') or '').startswith('interface') or target.get('action') == 'remove')
        if status == 'UNCERTAIN' and deterministic_patch and (not _repairable_representation_issue(verification)):
            break
        if status == 'UNCERTAIN' and _repairable_representation_issue(verification):
            working_contract = candidate
            working_verification = verification
        else:
            working_contract = deepcopy(last_repairable_contract)
            working_verification = deepcopy(last_repairable_verification)
    return {'resolved': False, 'accepted_contract': None, 'accepted_verification': None, 'attempts': history, 'attempts_used': len(history)}
repair_contract = repair_defective_contract
__all__ = ['REPAIR_VERSION', 'REPAIR_SYSTEM_PROMPT', 'choose_repair_target', 'repair_defective_contract', 'repair_contract']
