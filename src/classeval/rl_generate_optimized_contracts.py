from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.contract_clause_utils import merge_missing_clause_hints
from src.classeval.contract_normalizer import normalize_contract_shape, stage2_shape_errors
from src.classeval.contract_schema import new_contract_schema, new_method_schema
from src.classeval.core import (
    JsonDict,
    as_dict,
    as_list,
    class_name,
    compact_json,
    ensure_no_reference_leak,
    method_profiles,
    safe_methods_info,
    skeleton,
    task_id,
    text,
)
from src.classeval.evaluate_contract_guided import evaluation_path
from src.classeval.generate_contracts import TASK_FILE, contract_path as raw_contract_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.task_utils import select_tasks

STAGE = "2D_rl_optimized_contract_generation"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contracts"
POLICY_DIR = OUTPUT_ROOT / "classeval" / "rl_revision_policy"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contracts.log"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_DELAY = 0.0
DEFAULT_EPSILON = 0.15
DEFAULT_POLICY_TEMPERATURE = 1.0
DEFAULT_LR = 0.08
SYSTEM_PROMPT = "Return exactly one valid JSON object. No markdown, code, tests, or explanations."

ACTIONS: dict[str, str] = {
    "strengthen_postconditions": "Add precise return, state-update, output-format, and side-effect postconditions.",
    "add_edge_cases": "Add visible edge cases: empty inputs, missing values, duplicates, boundaries, repeated calls.",
    "clarify_interface": "Clarify parameter roles, return types, helper behavior, and instance/static behavior.",
    "add_invariants": "Add class-state invariants for fields preserved or modified across methods.",
    "remove_unsupported_constraints": "Remove vague, contradictory, unsupported, or over-strict clauses.",
    "failure_guided_repair": "Use raw failure feedback to strengthen the weakest contract clauses.",
}
ACTION_IDS = tuple(ACTIONS)

FEATURE_NAMES = (
    "bias",
    "raw_failed",
    "failure_logical",
    "failure_runtime",
    "failure_syntax",
    "failure_timeout",
    "failure_generation",
    "failure_unknown",
    "method_count",
    "field_count",
    "post_count",
    "edge_count",
)

logger = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def optimized_path(task: JsonDict, provider: str, model: str) -> Path:
    return OUT_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def policy_path(provider: str, model: str) -> Path:
    return POLICY_DIR / safe_name(provider) / safe_name(model) / "policy.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:16]


def short(value: Any, limit: int = 1200) -> str:
    value = text(value)
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def class_node(task: JsonDict) -> ast.ClassDef | None:
    try:
        tree = ast.parse(skeleton(task))
    except SyntaxError:
        return None
    return next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name(task)), None)


def function_header(source: str, node: ast.FunctionDef) -> str:
    segment = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    depth = 0
    quote: str | None = None
    escape = False

    for index, char in enumerate(segment):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ":" and depth == 0:
            return segment[:index].strip()

    return segment.splitlines()[0].strip()


def signature_map(task: JsonDict) -> dict[str, str]:
    source = skeleton(task)
    cls = class_node(task)
    if cls is None:
        return {}
    return {
        node.name: function_header(source, node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def static_map(task: JsonDict) -> dict[str, bool]:
    cls = class_node(task)
    if cls is None:
        return {}
    return {
        node.name: any(
            (isinstance(dec, ast.Name) and dec.id == "staticmethod")
            or (isinstance(dec, ast.Attribute) and dec.attr == "staticmethod")
            for dec in node.decorator_list
        )
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def expected_methods(task: JsonDict) -> list[str]:
    names = [
        text(info.get("method_name"))
        for info in safe_methods_info(task)
        if isinstance(info, dict) and text(info.get("method_name"))
    ]

    try:
        profiles = method_profiles(skeleton(task), class_name(task))
    except SyntaxError:
        profiles = {}

    for name in profiles:
        if name != "__init__" and name not in names:
            names.append(name)

    return names


def task_view(task: JsonDict) -> JsonDict:
    return {
        "task_id": task_id(task),
        "class_name": class_name(task),
        "class_description": text(task.get("class_description")),
        "class_constructor": text(task.get("class_constructor")),
        "fields": as_list(task.get("fields")),
        "methods_info": safe_methods_info(task),
        "skeleton": skeleton(task),
    }


def metadata_contract(task: JsonDict) -> JsonDict:
    sigs = signature_map(task)
    statics = static_map(task)
    contract = new_contract_schema()

    contract["task"].update(
        {
            "task_id": task_id(task),
            "benchmark": "ClassEval",
            "language": "python",
            "execution_model": "class_level",
            "class_name": class_name(task),
            "entry_point": class_name(task),
            "summary": text(task.get("class_description")),
        }
    )
    contract["class_interface"] = {
        "fields": as_list(task.get("fields")),
        "methods": expected_methods(task),
    }
    contract["constructor"]["signature"] = sigs.get("__init__", "")
    contract["constructor"]["initializes"] = as_list(task.get("fields"))
    contract["method_contracts"] = []

    for name in expected_methods(task):
        method = new_method_schema()
        method["method_name"] = name
        method["signature"] = sigs.get(name, "")
        method["is_static"] = bool(statics.get(name))
        contract["method_contracts"].append(method)

    return normalize_contract_shape(merge_missing_clause_hints(contract, task_view(task)))


def load_raw_contract_or_metadata(path: Path, task: JsonDict) -> tuple[JsonDict, str | None]:
    if not path.exists():
        return metadata_contract(task), f"Missing raw/v1 contract: {path}"

    try:
        record = load_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"Raw contract record must be a JSON object: {path}")
        if record.get("status") != "success":
            raise RuntimeError(text(record.get("error")) or f"Raw contract failed: {path}")
        if not isinstance(record.get("contract"), dict):
            raise ValueError(f"Raw contract record has no contract object: {path}")
        return normalize_contract_shape(record["contract"]), None
    except Exception as exc:
        logger.exception("Using metadata fallback contract for %s", task_id(task))
        return metadata_contract(task), str(exc)


def load_raw_feedback(task: JsonDict, provider: str, model: str) -> tuple[JsonDict | None, Path]:
    path = evaluation_path(task, provider, model, "raw")
    if path.exists():
        return load_json(path), path

    details_path = (
        RESULTS_ROOT
        / "classeval"
        / "contract_guided"
        / f"classeval_raw_contract_guided_{safe_name(provider)}_{safe_name(model)}_details.json"
    )
    if details_path.exists():
        for record in as_list(load_json(details_path)):
            if isinstance(record, dict) and record.get("task_id") == task_id(task):
                return record, details_path

    return None, path


def sanitize_feedback(value: Any) -> str:
    value = short(value, 1200)
    replacements = (
        (r'File ".*?", line \d+, in .*', ""),
        (r"Traceback \(most recent call last\):", ""),
        (r"self\.assert\w+\(.*", "assertion failed"),
        (r"AssertionError:.*", "assertion failed"),
        (r"Lists differ:.*", "generated output had wrong value/container format"),
        (r"First differing element.*", ""),
        (r"Ran \d+ tests? in .*", ""),
    )

    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)

    lines = []
    for line in value.splitlines():
        line = line.strip()
        if line and not line.startswith(("test_", "FAIL:", "ERROR:", "====", "----")):
            lines.append(line)

    return "\n".join(lines[:10])[:1200]


def feedback_summary(record: JsonDict | None, raw_contract_error: str | None) -> JsonDict:
    if not isinstance(record, dict):
        return {
            "available": False,
            "raw_contract_error": short(raw_contract_error, 500) if raw_contract_error else None,
        }

    metrics = as_dict(as_dict(record.get("evaluation")).get("metrics"))
    return {
        "available": True,
        "passed": record.get("passed") is True,
        "failure_type": text(record.get("failure_type")),
        "failure_stage": text(record.get("failure_stage")),
        "tests_total": metrics.get("total"),
        "tests_passed": metrics.get("passed"),
        "tests_failed": metrics.get("failed"),
        "failure_summary": sanitize_feedback(record.get("error") or record.get("failure_detail")),
        "raw_contract_error": short(raw_contract_error, 500) if raw_contract_error else None,
    }


def failure_bucket(feedback: JsonDict) -> str:
    failure = text(feedback.get("failure_type")).lower()
    stage = text(feedback.get("failure_stage")).lower()

    if "generation" in stage:
        return "generation"
    if "logical" in failure or "assert" in failure or "wrong" in failure:
        return "logical"
    if "runtime" in failure or "exception" in failure or "error" in failure:
        return "runtime"
    if "syntax" in failure or "indentation" in failure:
        return "syntax"
    if "timeout" in failure:
        return "timeout"

    return "unknown" if failure else ""


def contract_counts(contract: JsonDict) -> tuple[int, int]:
    post = edge = 0
    for method in as_list(contract.get("method_contracts")):
        body = as_dict(as_dict(method).get("contract"))
        post += len(as_list(body.get("postconditions")))
        edge += len(as_list(body.get("edge_cases")))
    return post, edge


def information_score(contract: JsonDict) -> int:
    score = len(as_list(contract.get("class_invariants"))) + len(as_list(contract.get("interaction_contracts")))
    for method in as_list(contract.get("method_contracts")):
        body = as_dict(as_dict(method).get("contract"))
        score += len(as_list(body.get("preconditions")))
        score += len(as_list(body.get("invariants")))
        score += 2 * len(as_list(body.get("postconditions")))
        score += 2 * len(as_list(body.get("edge_cases")))
    return score


def state_features(task: JsonDict, contract: JsonDict, feedback: JsonDict) -> JsonDict:
    method_count = max(1, len(expected_methods(task)))
    post_count, edge_count = contract_counts(contract)
    bucket = failure_bucket(feedback)

    features = {name: 0.0 for name in FEATURE_NAMES}
    features["bias"] = 1.0
    features["raw_failed"] = 1.0 if feedback.get("available") and feedback.get("passed") is not True else 0.0
    features["method_count"] = min(1.0, method_count / 12.0)
    features["field_count"] = min(1.0, len(as_list(task.get("fields"))) / 12.0)
    features["post_count"] = min(1.0, (post_count / method_count) / 6.0)
    features["edge_count"] = min(1.0, (edge_count / method_count) / 5.0)

    key = f"failure_{bucket}"
    if key in features:
        features[key] = 1.0

    return features


def fresh_policy() -> JsonDict:
    return {
        "stage": "2D_contextual_bandit_policy",
        "policy_type": "linear_epsilon_greedy",
        "feature_names": list(FEATURE_NAMES),
        "actions": ACTIONS,
        "weights": {action: {name: 0.0 for name in FEATURE_NAMES} for action in ACTION_IDS},
        "reward_count": 0,
        "reward_mean": 0.0,
        "reward_history": [],
        "updated_at": now(),
    }


def load_policy(path: Path) -> JsonDict:
    policy = fresh_policy()

    if path.exists():
        value = load_json(path)
        if isinstance(value, dict):
            policy.update(value)

    weights = as_dict(policy.get("weights"))
    policy["weights"] = {
        action: {
            name: float(as_dict(weights.get(action)).get(name, 0.0))
            for name in FEATURE_NAMES
        }
        for action in ACTION_IDS
    }
    policy["actions"] = ACTIONS
    policy["feature_names"] = list(FEATURE_NAMES)
    return policy


def score(policy: JsonDict, action: str, features: JsonDict) -> float:
    weights = as_dict(policy["weights"][action])
    return sum(float(weights.get(name, 0.0)) * float(features.get(name, 0.0)) for name in FEATURE_NAMES)


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    top = max(scores.values()) if scores else 0.0
    exps = {
        key: math.exp((value - top) / max(1e-6, temperature))
        for key, value in scores.items()
    }
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def select_action(policy: JsonDict, features: JsonDict, args: argparse.Namespace) -> JsonDict:
    seed = args.seed + int(hashlib.sha256(canonical(features).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    scores = {action: score(policy, action, features) for action in ACTION_IDS}

    if args.greedy_action:
        action, mode, explored = max(scores, key=scores.get), "greedy", False
    elif rng.random() < args.epsilon:
        action, mode, explored = rng.choice(list(ACTION_IDS)), "epsilon_random", True
    else:
        action, mode, explored = max(scores, key=scores.get), "epsilon_greedy", False

    return {
        "action_id": action,
        "selected_action": {"id": action, "instruction": ACTIONS[action]},
        "selection_mode": mode,
        "scores": scores,
        "probabilities": softmax(scores, args.policy_temperature),
        "explored": explored,
    }


def reward_from_pair(raw_eval: JsonDict | None, opt_eval: JsonDict | None) -> float:
    if not isinstance(raw_eval, dict) or not isinstance(opt_eval, dict):
        return 0.0

    raw_passed = raw_eval.get("passed") is True
    opt_passed = opt_eval.get("passed") is True

    if raw_passed and opt_passed:
        return 0.2
    if raw_passed and not opt_passed:
        return -1.0
    if not raw_passed and opt_passed:
        return 1.0
    return -0.2


def update_policy(policy: JsonDict, action: str, features: JsonDict, reward: float, lr: float) -> None:
    if action not in ACTION_IDS:
        return

    count = int(policy.get("reward_count") or 0)
    mean = float(policy.get("reward_mean") or 0.0)
    advantage = reward - mean

    for name in FEATURE_NAMES:
        policy["weights"][action][name] += lr * advantage * float(features.get(name, 0.0))

    policy["reward_count"] = count + 1
    policy["reward_mean"] = mean + (reward - mean) / float(count + 1)
    policy["updated_at"] = now()


def update_policy_from_previous_results(tasks: list[JsonDict], args: argparse.Namespace, model: str, policy: JsonDict) -> int:
    seen = {
        item.get("key")
        for item in as_list(policy.get("reward_history"))
        if isinstance(item, dict)
    }
    updates = 0
    base_provider = args.base_provider or args.provider
    base_model = args.base_model or model

    for task in tasks:
        opt_path = optimized_path(task, args.provider, model)
        opt_eval_path = evaluation_path(task, args.provider, model, "optimized_rl")
        raw_eval_path = evaluation_path(task, base_provider, base_model, "raw")

        if not (opt_path.exists() and opt_eval_path.exists() and raw_eval_path.exists()):
            continue

        opt_record = load_json(opt_path)
        rl = as_dict(opt_record.get("rl_policy"))
        action = text(rl.get("action_id"))
        features = as_dict(rl.get("state_features"))

        if action not in ACTION_IDS or not features:
            continue

        opt_eval = load_json(opt_eval_path)
        raw_eval = load_json(raw_eval_path)
        key = "|".join([task_id(task), action, text(opt_record.get("contract_hash")), text(opt_eval.get("passed"))])

        if key in seen:
            continue

        reward = reward_from_pair(raw_eval, opt_eval)
        update_policy(policy, action, features, reward, args.learning_rate)
        policy["reward_history"].append(
            {"key": key, "task_id": task_id(task), "action": action, "reward": reward}
        )
        seen.add(key)
        updates += 1

    return updates


def merge_candidate(candidate: JsonDict, raw_contract: JsonDict, task: JsonDict) -> JsonDict:
    raw = normalize_contract_shape(raw_contract)
    body = candidate.get("contract") if isinstance(candidate.get("contract"), dict) else candidate
    contract = normalize_contract_shape(body)

    contract["task"] = raw["task"]
    contract["class_interface"] = raw["class_interface"]
    contract["constructor"]["signature"] = raw["constructor"].get("signature")
    contract["constructor"]["initializes"] = raw["constructor"].get("initializes")

    raw_methods = {
        text(method.get("method_name")): method
        for method in as_list(raw.get("method_contracts"))
        if isinstance(method, dict)
    }
    new_methods = {
        text(method.get("method_name")): method
        for method in as_list(contract.get("method_contracts"))
        if isinstance(method, dict)
    }

    merged = []
    for name in expected_methods(task):
        base = as_dict(raw_methods.get(name)).copy()
        current = as_dict(new_methods.get(name))
        if current:
            base["dependencies"] = current.get("dependencies") or base.get("dependencies")
            base["contract"] = current.get("contract") or base.get("contract")
        merged.append(base)

    contract["method_contracts"] = merged
    return normalize_contract_shape(merge_missing_clause_hints(contract, task_view(task)))


def parse_candidate(response: str, raw_contract: JsonDict, task: JsonDict) -> JsonDict:
    parsed = extract_json_object(response)
    if not isinstance(parsed, dict):
        raise ValueError("Model response did not contain a JSON object.")

    contract = merge_candidate(parsed, raw_contract, task)
    errors = stage2_shape_errors(contract, expected_methods=expected_methods(task))
    if errors:
        raise ValueError("Stage 2 shape errors: " + "; ".join(errors[:10]))

    return contract


def build_prompt(task: JsonDict, raw_contract: JsonDict, feedback: JsonDict, selection: JsonDict) -> str:
    payload = {
        "selected_rl_action": selection["selected_action"],
        "visible_task": task_view(task),
        "raw_evaluation_feedback": feedback,
        "raw_contract": raw_contract,
    }
    prompt = "\n".join(
        [
            "Optimize the Stage-2 ClassEval contract using the selected contextual-bandit action.",
            "Use only visible task data, raw contract, and summarized raw-evaluation feedback.",
            "Do not use reference solutions, hidden tests, exact test answers, code, markdown, or prose.",
            "Preserve class name, fields, constructor, method names, signatures, and staticmethod status exactly.",
            "Return the full optimized contract JSON object.",
            compact_json(payload),
        ]
    )
    ensure_no_reference_leak(task, prompt)
    return prompt


def make_record(task: JsonDict, args: argparse.Namespace, model: str, **extra: Any) -> JsonDict:
    return {
        "task_id": task_id(task),
        "benchmark": "ClassEval",
        "dataset": "classeval",
        "execution_model": "class_level",
        "class_name": class_name(task),
        "entry_point": class_name(task),
        "stage": STAGE,
        "provider": args.provider,
        "model_name": model,
        "base_provider": args.base_provider or args.provider,
        "base_model": args.base_model or model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "created_at": now(),
        **extra,
    }


def success_record(
    task: JsonDict,
    args: argparse.Namespace,
    model: str,
    contract: JsonDict,
    raw_path: Path,
    raw_eval_path: Path,
    status: str,
    features: JsonDict,
    raw_error: str | None,
    **extra: Any,
) -> JsonDict:
    return make_record(
        task,
        args,
        model,
        status="success",
        contract=contract,
        candidate_contract=extra.pop("candidate_contract", None),
        contract_type="rl_optimized",
        contract_version="v2_rl_optimized",
        optimization_status=status,
        previous_contract_file=str(raw_path),
        previous_evaluation_file=str(raw_eval_path),
        raw_contract_error=raw_error,
        contract_hash=content_hash(contract),
        rl_policy=extra.pop("rl_policy", {"state_features": features, "llm_called": False}),
        error=extra.pop("error", None),
        **extra,
    )


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any, policy: JsonDict) -> JsonDict:
    base_provider = args.base_provider or args.provider
    base_model = args.base_model or model

    raw_path = raw_contract_path(task, base_provider, base_model)
    raw_contract, raw_error = load_raw_contract_or_metadata(raw_path, task)
    raw_eval, raw_eval_path = load_raw_feedback(task, base_provider, base_model)

    feedback = feedback_summary(raw_eval, raw_error)
    features = state_features(task, raw_contract, feedback)

    if feedback.get("available") is not True:
        reason = "Missing raw/v1 evaluation feedback; preserved raw contract as explicit optimized-path fallback."
        return success_record(
            task,
            args,
            model,
            raw_contract,
            raw_path,
            raw_eval_path,
            "missing_feedback_raw_fallback",
            features,
            raw_error,
            optimization_changed=False,
            optimized_success=False,
            fallback_used=True,
            issue="missing_raw_feedback",
            reason=reason,
            solution="Run raw contract-guided evaluation before RL optimization to enable contextual-bandit action selection.",
            feedback=feedback,
            raw_information_score=information_score(raw_contract),
            optimized_information_score=information_score(raw_contract),
            rl_policy={
                "action_id": "raw_fallback_missing_feedback",
                "selected_action": {
                    "id": "raw_fallback_missing_feedback",
                    "instruction": "Raw feedback unavailable.",
                },
                "state_features": features,
                "llm_called": False,
            },
            error=reason,
        )

    if feedback.get("passed") is True:
        return success_record(
            task,
            args,
            model,
            raw_contract,
            raw_path,
            raw_eval_path,
            "preserved_raw_passed",
            features,
            raw_error,
            optimization_changed=False,
            optimized_success=False,
            fallback_used=False,
            raw_information_score=information_score(raw_contract),
            optimized_information_score=information_score(raw_contract),
            rl_policy={
                "action_id": "keep_raw_if_passed",
                "selected_action": {
                    "id": "keep_raw_if_passed",
                    "instruction": "Raw/v1 passed; preserve raw contract.",
                },
                "state_features": features,
                "llm_called": False,
            },
        )

    selection = select_action(policy, features, args)
    prompt = build_prompt(task, raw_contract, feedback, selection)
    candidate: JsonDict | None = None

    try:
        response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=True,
        )

        candidate = parse_candidate(response, raw_contract, task)
        if canonical(candidate) == canonical(raw_contract):
            raise ValueError("Candidate contract was identical to raw.")

        return success_record(
            task,
            args,
            model,
            candidate,
            raw_path,
            raw_eval_path,
            "optimized",
            features,
            raw_error,
            candidate_contract=candidate,
            optimization_changed=True,
            optimized_success=True,
            fallback_used=False,
            feedback=feedback,
            raw_information_score=information_score(raw_contract),
            optimized_information_score=information_score(candidate),
            rl_policy={**selection, "state_features": features, "llm_called": True},
            raw_response=response,
            api_result=api_result,
            api_latency_seconds=api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
        )

    except Exception as exc:
        logger.exception("Optimized contract generation failed for %s", task_id(task))
        return success_record(
            task,
            args,
            model,
            raw_contract,
            raw_path,
            raw_eval_path,
            "no_optimized_candidate",
            features,
            raw_error,
            candidate_contract=candidate,
            optimization_changed=False,
            optimized_success=False,
            fallback_used=True,
            issue="no_optimized_candidate",
            reason=short(exc),
            solution="Raw fallback preserved; do not count this as optimized success.",
            feedback=feedback,
            raw_information_score=information_score(raw_contract),
            optimized_information_score=information_score(candidate or raw_contract),
            rl_policy={**selection, "state_features": features, "llm_called": True},
            error=short(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ClassEval optimized contracts with contextual-bandit RL")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-provider")
    parser.add_argument("--base-model")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--policy-temperature", type=float, default=DEFAULT_POLICY_TEMPERATURE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--greedy-action", action="store_true")
    parser.add_argument("--no-policy-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    ppath = policy_path(args.provider, model)
    policy = load_policy(ppath)

    updates = 0
    if not args.no_policy_update:
        updates = update_policy_from_previous_results(tasks, args, model, policy)
        save_json(ppath, policy)

    counts = {"optimized": 0, "preserved": 0, "fallback": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = optimized_path(task, args.provider, model)

        if path.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        record = generate_one(task, args, model, client, policy)
        save_json(path, record)

        status = record.get("optimization_status")
        if record["status"] != "success":
            counts["failed"] += 1
            print(f"[{index}] FAIL {task_id(task)}: {record.get('error')}")
        elif status == "optimized":
            counts["optimized"] += 1
            print(f"[{index}] OPTIMIZED {task_id(task)} action={(record.get('rl_policy') or {}).get('action_id')}")
        elif status == "preserved_raw_passed":
            counts["preserved"] += 1
            print(f"[{index}] KEEP {task_id(task)}")
        else:
            counts["fallback"] += 1
            print(f"[{index}] FALLBACK {task_id(task)}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval optimized contract generation finished.")
    print(json.dumps(counts, indent=2))
    print(f"Policy updates from previous optimized evaluations: {updates}")
    print(f"Policy: {ppath}")
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


if __name__ == "__main__":
    main()