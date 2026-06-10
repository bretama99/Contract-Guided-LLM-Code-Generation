#!/usr/bin/env python3
from __future__ import annotations
from src.evalplus_integration.paths import results_folder as evalplus_results_folder
import argparse
import hashlib
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import DATASETS, LOG_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.parsing import extract_json_object
from src.common.raw_contract_paths import raw_contract_path, raw_contract_results_folder
from src.common.task_utils import select_tasks, task_entry_point, task_identifier, task_prompt
from src.contract_synthesis.contract_normalizer import normalize_contract
from src.contract_synthesis.contract_schema import make_contract_schema_for_task
from src.contract_synthesis.contract_schema_constants import SCHEMA_VERSION as RAW_SCHEMA_VERSION

STAGE = "stage_2d_rl_optimized_contract_synthesis"
METHOD = "optimized_rl_contract_generation"
OPT_SCHEMA_VERSION = "stage2_rl_optimized_contract_schema_v1"

LOG_FILE = LOG_ROOT / "stage2_rl_optimized_contracts.log"
OPT_CODE_ROOT = OUTPUT_ROOT / "contract_guided_generation" / "optimized_rl"
OPT_RESULT_ROOT = RESULTS_ROOT / "contract_guided_generation" / "optimized_rl"
POLICY_ROOT = RESULTS_ROOT / "rl_method_level" / "policies"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 3000
DEFAULT_DELAY = 0.0
DEFAULT_LR = 0.08
DEFAULT_EPSILON = 0.2
DEFAULT_SOFTMAX_TEMPERATURE = 1.0

SYSTEM_PROMPT = (
    "You are an RL-selected contract optimizer for Python benchmark tasks. "
    "Return exactly one valid JSON contract object. Do not return code, tests, markdown, or explanations."
)

ACTIONS: dict[str, str] = {
    "keep_raw_if_passed": (
        "The raw/v1 contract-guided code already passed evaluation. Preserve the raw contract unchanged."
    ),
    "repair_failure_specific": (
        "Revise the contract to address the previous failure type and failure explanation. Add concrete, "
        "testable behavioral clauses that directly prevent the observed failure."
    ),
    "strengthen_postconditions": (
        "Make postconditions more precise, observable, and testable while preserving only prompt-supported behavior."
    ),
    "add_edge_cases": (
        "Add valid edge cases and expected behavior clearly supported or strongly implied by the task prompt."
    ),
    "clarify_interface": (
        "Clarify input/output descriptions, parameter roles, return behavior, imports, and helper functions without "
        "changing the required signature."
    ),
    "remove_unsupported_constraints": (
        "Remove or weaken constraints not supported by the prompt, especially invented non-empty, sorted, unique, "
        "positive, non-null, or exception requirements."
    ),
}

ACTION_ORDER = tuple(ACTIONS)
REVISION_ACTIONS = tuple(a for a in ACTION_ORDER if a != "keep_raw_if_passed")

FEATURE_NAMES = (
    "bias",
    "raw_passed",
    "raw_failed",
    "feedback_missing",
    "failure_logical",
    "failure_runtime",
    "failure_syntax",
    "failure_timeout",
    "failure_signature",
    "failure_import",
    "failure_missing_method",
    "failure_other",
    "pre_count",
    "post_count",
    "inv_count",
    "edge_count",
)

FAILURE_REWARD = {
    None: 1.0,
    "logical": -0.30,
    "wrong_answer": -0.30,
    "runtime": -0.70,
    "timeout": -0.80,
    "syntax": -1.00,
    "import": -0.95,
    "signature": -0.95,
    "missing_method": -0.95,
    "other": -0.60,
}

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def short(value: Any, limit: int = 1200) -> str:
    text = "" if value is None else str(value).strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()[:16]


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_failure(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "logical_error": "logical",
        "assertion": "logical",
        "assertion_error": "logical",
        "wrong_answer": "logical",
        "runtime_error": "runtime",
        "syntax_error": "syntax",
        "import_error": "import",
        "signature_error": "signature",
        "method_missing": "missing_method",
    }
    return aliases.get(text, text)


def clause_count(contract: dict[str, Any], key: str) -> int:
    return len(as_list(contract.get(key)))


def contract_counts(contract: dict[str, Any]) -> dict[str, int]:
    return {
        "preconditions": clause_count(contract, "preconditions"),
        "postconditions": clause_count(contract, "postconditions"),
        "invariants": clause_count(contract, "invariants"),
        "edge_cases": clause_count(contract, "edge_cases"),
    }


# ---------------------------------------------------------------------------
# Paths. Optimized contract files are stored next to raw contract files.
# ---------------------------------------------------------------------------

def optimized_contract_folder(dataset: str, provider: str, model: str) -> Path:
    probe = raw_contract_path(dataset, "__probe__", provider, model)
    return probe.parent


def optimized_contract_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return optimized_contract_folder(dataset, provider, model) / f"{safe_name(task_id)}_optimized_contract.json"


def optimized_contract_code_folder(dataset: str, provider: str, model: str) -> Path:
    return OPT_CODE_ROOT / safe_name(provider) / safe_name(model) / safe_name(dataset)


def optimized_contract_code_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return optimized_contract_code_folder(dataset, provider, model) / f"{safe_name(task_id)}_optimized_contract_guided.json"


def optimized_contract_results_folder(dataset: str, provider: str, model: str) -> Path:
    return OPT_RESULT_ROOT / safe_name(provider) / safe_name(model) / safe_name(dataset)


def policy_path(dataset: str, provider: str, model: str) -> Path:
    return POLICY_ROOT / safe_name(provider) / safe_name(model) / safe_name(dataset) / "policy.json"


# ---------------------------------------------------------------------------
# Loading contracts/evaluation feedback.
# ---------------------------------------------------------------------------

def extract_contract_record(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} contract record must be a JSON object.")
    if record.get("status") not in (None, "success"):
        raise RuntimeError(f"{label} contract status is not success: {record.get('error')}")
    contract = record.get("contract", record)
    if not isinstance(contract, dict):
        raise ValueError(f"{label} contract record does not contain a contract object.")
    return contract


def try_load_raw_contract(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"Missing raw/v1 contract: {path}"
    try:
        return extract_contract_record(load_json(path), "raw/v1"), None
    except Exception as exc:
        return None, str(exc)


def load_evaluation_rows(folder: Path) -> list[dict[str, Any]]:
    if not folder.exists():
        return []

    details_path = folder / "details.json"
    if details_path.exists():
        try:
            data = load_json(details_path)
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            if isinstance(data, dict) and isinstance(data.get("details"), list):
                return [row for row in data["details"] if isinstance(row, dict)]
        except Exception:
            logger.exception("Could not load evaluation details: %s", details_path)

    rows: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("*.json")):
        if path.name in {"summary.json", "export_summary.json", "policy.json"}:
            continue
        try:
            row = load_json(path)
        except Exception:
            continue
        if isinstance(row, dict) and row.get("task_id"):
            rows.append(row)
    return rows

def feedback_results_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    if "evalplus" in DATASETS.get(dataset, {}):
        return evalplus_results_folder(method, dataset, provider, model)

    if method == "raw_contracts":
        return raw_contract_results_folder(dataset, provider, model)

    if method == "optimized_rl":
        return optimized_contract_results_folder(dataset, provider, model)

    raise ValueError(f"Unsupported feedback method: {method}")
def evaluation_map(folder: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in load_evaluation_rows(folder) if row.get("task_id")}

def make_feedback(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"available": False}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    return {
        "available": True,
        "passed": row.get("passed") is True,
        "failure_type": normalize_failure(row.get("failure_type")),
        "failure_explanation": short(row.get("failure_explanation"), 1000),
        "evaluation_error": short(evaluation.get("error"), 1000),
    }


def diagnose_feedback(feedback: dict[str, Any], *, raw_contract_error: str | None = None) -> dict[str, str]:
    if raw_contract_error:
        return {
            "issue": "invalid_raw_contract",
            "reason": f"The raw/v1 contract artifact could not be loaded: {short(raw_contract_error, 300)}",
            "solution": "Recover by generating a valid optimized contract from the task prompt, then rerun optimized code generation and evaluation.",
        }
    if feedback.get("passed") is True:
        return {
            "issue": "raw_passed_preserved",
            "reason": "Raw/v1 contract-guided code already passed evaluation.",
            "solution": "Preserve the raw contract and reuse the raw generated code; do not rewrite or regenerate.",
        }

    failure = normalize_failure(feedback.get("failure_type")) or "other"
    if failure == "logical":
        return {
            "issue": "semantic_contract_insufficient",
            "reason": "The generated code failed assertions, so the contract likely missed an exact semantic requirement or edge case.",
            "solution": "Add concrete postconditions and edge cases that directly encode the failed behavior and expected output relation.",
        }
    if failure == "runtime":
        return {
            "issue": "runtime_edge_case_or_interface_gap",
            "reason": "The generated code crashed on a valid benchmark input.",
            "solution": "Clarify supported input shapes and edge cases so generated code does not assume invalid lengths, types, or non-empty data.",
        }
    if failure == "timeout":
        return {
            "issue": "algorithmic_efficiency_gap",
            "reason": "The generated code timed out, likely because the contract did not discourage an inefficient strategy.",
            "solution": "Add a prompt-supported efficiency/strategy clause, such as generating required sequences directly rather than brute-force scanning.",
        }
    if failure in {"syntax", "import", "signature", "missing_method"}:
        return {
            "issue": "interface_or_generation_format_gap",
            "reason": "Generation failed before semantic evaluation because the interface, imports, or output format was wrong.",
            "solution": "Clarify required signature, imports, helper functions, and output-only-code constraints in the optimized contract.",
        }
    return {
        "issue": "unknown_generation_or_evaluation_failure",
        "reason": "The previous evaluation failed, but the failure type was not specific enough for a narrower diagnosis.",
        "solution": "Strengthen observable postconditions and add edge cases derived from the prompt and failure explanation.",
    }


# ---------------------------------------------------------------------------
# Contextual bandit policy.
# ---------------------------------------------------------------------------

def empty_weights() -> dict[str, dict[str, float]]:
    return {action: {name: 0.0 for name in FEATURE_NAMES} for action in ACTION_ORDER}


def new_policy() -> dict[str, Any]:
    return {
        "policy_version": "contextual_bandit_linear_softmax_v2",
        "stage": STAGE,
        "actions": list(ACTION_ORDER),
        "feature_names": list(FEATURE_NAMES),
        "weights": empty_weights(),
        "reward_count": 0,
        "reward_mean": 0.0,
        "reward_history": [],
        "updated_at": utc_now(),
    }


def ensure_policy_shape(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        policy = new_policy()
    policy.setdefault("policy_version", "contextual_bandit_linear_softmax_v2")
    policy.setdefault("stage", STAGE)
    policy.setdefault("actions", list(ACTION_ORDER))
    policy.setdefault("feature_names", list(FEATURE_NAMES))
    policy.setdefault("weights", {})
    policy.setdefault("reward_count", 0)
    policy.setdefault("reward_mean", 0.0)
    policy.setdefault("reward_history", [])
    if not isinstance(policy["weights"], dict):
        policy["weights"] = {}
    for action in ACTION_ORDER:
        policy["weights"].setdefault(action, {})
        for name in FEATURE_NAMES:
            policy["weights"][action].setdefault(name, 0.0)
    policy["actions"] = list(ACTION_ORDER)
    policy["feature_names"] = list(FEATURE_NAMES)
    return policy

def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_policy()
    try:
        return ensure_policy_shape(load_json(path))
    except Exception:
        logger.exception("Could not load policy, starting fresh: %s", path)
        return new_policy()


def score_action(policy: dict[str, Any], action: str, features: dict[str, float]) -> float:
    weights = policy["weights"][action]
    return sum(float(weights.get(name, 0.0)) * float(value) for name, value in features.items())


def action_scores(policy: dict[str, Any], features: dict[str, float], actions: tuple[str, ...]) -> dict[str, float]:
    return {action: score_action(policy, action, features) for action in actions}


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    temperature = max(float(temperature), 1e-6)
    scaled = {key: value / temperature for key, value in scores.items()}
    max_score = max(scaled.values()) if scaled else 0.0
    exps = {key: math.exp(value - max_score) for key, value in scaled.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def sample_from_probs(probs: dict[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for action, prob in probs.items():
        cumulative += prob
        if threshold <= cumulative:
            return action
    return next(reversed(probs))


def select_action(
    *,
    policy: dict[str, Any],
    features: dict[str, float],
    raw_passed: bool,
    mode: str,
    epsilon: float,
    temperature: float,
    rng: random.Random,
) -> tuple[str, dict[str, Any]]:
    if raw_passed:
        return "keep_raw_if_passed", {
            "mode": "forced",
            "reason": "raw_contract_guided_evaluation_passed",
            "scores": {},
            "probabilities": {},
        }

    scores = action_scores(policy, features, REVISION_ACTIONS)
    probs = softmax(scores, temperature)
    greedy = max(scores, key=scores.get)

    if mode == "greedy":
        action, reason = greedy, "greedy_best_score"
    elif mode == "softmax":
        action, reason = sample_from_probs(probs, rng), "softmax_sample"
    elif mode == "random":
        action, reason = rng.choice(list(REVISION_ACTIONS)), "uniform_random"
    else:
        if rng.random() < epsilon:
            action, reason = rng.choice(list(REVISION_ACTIONS)), "epsilon_random_exploration"
        else:
            action, reason = greedy, "epsilon_greedy_exploitation"

    return action, {
        "mode": mode,
        "reason": reason,
        "epsilon": epsilon,
        "softmax_temperature": temperature,
        "scores": scores,
        "probabilities": probs,
    }


def build_state(task: dict[str, Any], raw_contract: dict[str, Any] | None, feedback: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_identifier(task),
        "entry_point": task_entry_point(task),
        "raw_contract_counts": contract_counts(raw_contract or {}),
        "raw_feedback": feedback,
    }


def features_from_state(state: dict[str, Any]) -> dict[str, float]:
    feedback = state.get("raw_feedback") if isinstance(state.get("raw_feedback"), dict) else {}
    counts = state.get("raw_contract_counts") if isinstance(state.get("raw_contract_counts"), dict) else {}
    failure = normalize_failure(feedback.get("failure_type")) or "none"

    features = {name: 0.0 for name in FEATURE_NAMES}
    features["bias"] = 1.0
    features["feedback_missing"] = 0.0 if feedback.get("available") else 1.0
    features["raw_passed"] = 1.0 if feedback.get("passed") is True else 0.0
    features["raw_failed"] = 1.0 if feedback.get("available") and feedback.get("passed") is not True else 0.0

    failure_feature = f"failure_{failure}"
    if failure_feature in features:
        features[failure_feature] = 1.0
    elif failure != "none":
        features["failure_other"] = 1.0

    features["pre_count"] = min(float(counts.get("preconditions", 0)) / 10.0, 1.0)
    features["post_count"] = min(float(counts.get("postconditions", 0)) / 10.0, 1.0)
    features["inv_count"] = min(float(counts.get("invariants", 0)) / 10.0, 1.0)
    features["edge_count"] = min(float(counts.get("edge_cases", 0)) / 10.0, 1.0)
    return features

# ---------------------------------------------------------------------------
# Policy reward update.
# ---------------------------------------------------------------------------

def reward_from_result(row: dict[str, Any]) -> float:
    if row.get("passed") is True:
        return 1.0
    return FAILURE_REWARD.get(normalize_failure(row.get("failure_type")), FAILURE_REWARD["other"])


def reward_from_raw_and_optimized(raw_feedback: dict[str, Any], opt_row: dict[str, Any], opt_record: dict[str, Any]) -> float:
    raw_passed = raw_feedback.get("passed") if raw_feedback.get("available") else None
    opt_passed = opt_row.get("passed") is True
    status = opt_record.get("optimization_status")

    if raw_passed is True:
        return 0.60 if opt_passed else -1.00
    if raw_passed is False and opt_passed:
        return 1.00
    if raw_passed is False and status == "no_optimized_candidate":
        return -0.35
    if raw_passed is False:
        return min(reward_from_result(opt_row), -0.25)
    return reward_from_result(opt_row)


def apply_policy_update(
    *,
    policy: dict[str, Any],
    selected_action: str,
    features: dict[str, float],
    reward: float,
    learning_rate: float,
    selection_temperature: float,
) -> None:
    if selected_action not in ACTIONS:
        return
    count = int(policy.get("reward_count", 0))
    mean = float(policy.get("reward_mean", 0.0))
    advantage = reward - mean
    scores = action_scores(policy, features, ACTION_ORDER)
    probs = softmax(scores, selection_temperature)

    for action in ACTION_ORDER:
        grad = (1.0 if action == selected_action else 0.0) - probs.get(action, 0.0)
        for name, value in features.items():
            policy["weights"][action][name] += learning_rate * advantage * grad * value

    policy["reward_count"] = count + 1
    policy["reward_mean"] = mean + ((reward - mean) / float(count + 1))


def load_optimized_contract_records(folder: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not folder.exists():
        return records
    for path in sorted(folder.glob("*_optimized_contract.json")):
        try:
            record = load_json(path)
        except Exception:
            continue
        if isinstance(record, dict) and record.get("task_id"):
            item = dict(record)
            item["path"] = str(path)
            records[str(record["task_id"])] = item
    return records


def reward_key(row: dict[str, Any], record: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("task_id")),
            str(record.get("rl", {}).get("selected_action")),
            str(record.get("contract_hash") or content_hash(record.get("contract"))),
            str(row.get("passed")),
            str(row.get("failure_type")),
        ]
    )


def update_policy_from_optimized_results(
    *,
    policy: dict[str, Any],
    dataset: str,
    provider: str,
    model: str,
    raw_feedback_by_task: dict[str, dict[str, Any]],
    learning_rate: float,
    selection_temperature: float,
) -> int:
    rows = load_evaluation_rows(
    feedback_results_folder("optimized_rl", dataset, provider, model)
)
    records = load_optimized_contract_records(optimized_contract_folder(dataset, provider, model))
    processed = {
        item.get("key")
        for item in as_list(policy.get("reward_history"))
        if isinstance(item, dict) and item.get("key")
    }

    updates = 0
    for row in rows:
        task_id = str(row.get("task_id") or "")
        record = records.get(task_id)
        if not record:
            continue
        rl = record.get("rl") if isinstance(record.get("rl"), dict) else {}
        selected_action = rl.get("selected_action")
        features = rl.get("state_features")
        if selected_action not in ACTIONS or not isinstance(features, dict):
            continue
        key = reward_key(row, record)
        if key in processed:
            continue

        raw_feedback = make_feedback(raw_feedback_by_task.get(task_id))
        reward = reward_from_raw_and_optimized(raw_feedback, row, record)
        numeric_features = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
        apply_policy_update(
            policy=policy,
            selected_action=selected_action,
            features=numeric_features,
            reward=reward,
            learning_rate=learning_rate,
            selection_temperature=selection_temperature,
        )
        policy["reward_history"].append(
            {
                "key": key,
                "task_id": task_id,
                "selected_action": selected_action,
                "reward": reward,
                "raw_passed": raw_feedback.get("passed"),
                "optimized_passed": row.get("passed") is True,
                "optimization_status": record.get("optimization_status"),
                "failure_type": normalize_failure(row.get("failure_type")),
                "updated_at": utc_now(),
            }
        )
        processed.add(key)
        updates += 1
    policy["updated_at"] = utc_now()
    return updates


# ---------------------------------------------------------------------------
# Contract generation.
# ---------------------------------------------------------------------------

def build_revision_prompt(
    *,
    task: dict[str, Any],
    raw_contract: dict[str, Any],
    feedback: dict[str, Any],
    selected_action: str,
) -> str:
    diagnosis = diagnose_feedback(feedback)
    return "\n".join(
        [
            "Revise the v1/raw Design-by-Contract specification into one optimized Stage 2 contract.",
            "",
            "Hard rules:",
            "- Return exactly one valid JSON object.",
            "- Keep the same overall schema shape as the v1/raw contract.",
            "- Use only the original prompt, signature, docstring, examples, and previous raw evaluation feedback.",
            "- Do not use hidden tests, canonical solutions, generated code, or test answers.",
            "- Do not generate Python code.",
            "- Do not add input validation, exceptions, or constraints unless explicitly supported.",
            "- Because raw/v1 evaluation failed, the optimized contract should change at least one useful clause.",
            "- Add at least one concrete postcondition or edge case that directly addresses the failure.",
            "- This is a single Stage 2 optimization decision; do not rely on iterative retry or repair.",
            "",
            "RL-selected revision action:",
            selected_action,
            ACTIONS[selected_action],
            "",
            "Task ID:",
            task_identifier(task),
            "",
            "Entry point:",
            task_entry_point(task),
            "",
            "Original benchmark prompt:",
            task_prompt(task),
            "",
            "v1/raw contract:",
            json.dumps(raw_contract, indent=2, ensure_ascii=False),
        ]
    )
    
def build_recovery_prompt(*, task: dict[str, Any], dataset: str, raw_error: str, feedback: dict[str, Any]) -> str:
    schema = make_contract_schema_for_task(task, dataset)
    diagnosis = diagnose_feedback(feedback, raw_contract_error=raw_error)
    return "\n".join(
        [
            "The raw/v1 contract artifact is invalid or missing. Generate a valid optimized Stage 2 contract directly from the task prompt.",
            "Return exactly one valid JSON object only. Do not return code, tests, markdown, or explanations.",
            "Use only the visible benchmark prompt, signature, docstring, and examples.",
            "",
            "Failure diagnosis:",
            json.dumps(diagnosis, indent=2, ensure_ascii=False),
            "",
            "Required schema shape:",
            json.dumps(schema, indent=2, ensure_ascii=False),
            "",
            "Task ID:",
            task_identifier(task),
            "",
            "Entry point:",
            task_entry_point(task),
            "",
            "Original benchmark prompt:",
            task_prompt(task),
            "",
            "Raw contract error:",
            raw_error,
        ]
    )


def parse_contract_response(raw_response: str) -> dict[str, Any]:
    parsed = extract_json_object(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Optimized contract response must be a JSON object.")
    return parsed


def classify_candidate(raw_contract: dict[str, Any], candidate: dict[str, Any], feedback: dict[str, Any]) -> tuple[str, bool]:
    changed = stable_json(candidate) != stable_json(raw_contract)
    if changed:
        return "optimized", True
    if feedback.get("passed") is True:
        return "preserved_raw_passed", False
    return "no_optimized_candidate", False


def optimized_success_for(status: str | None) -> bool:
    return status in {"optimized", "preserved_raw_passed"}


def make_record(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    raw_provider: str,
    raw_model: str,
    temperature: float,
    max_tokens: int,
    status: str,
    raw_contract_file: Path | None,
    contract: dict[str, Any] | None,
    candidate_contract: dict[str, Any] | None,
    prompt: str,
    raw_response: str,
    api_result: dict[str, Any] | None,
    error: str | None,
    rl: dict[str, Any] | None,
    issue: str | None = None,
    reason: str | None = None,
    solution: str | None = None,
) -> dict[str, Any]:
    optimization_status = None
    optimization_changed = None
    selected_action = None
    if isinstance(rl, dict):
        optimization_status = rl.get("optimization_status")
        optimization_changed = rl.get("optimization_changed")
        selected_action = rl.get("selected_action")

    return {
        "task_id": task_identifier(task),
        "benchmark": benchmark,
        "dataset": dataset,
        "entry_point": task_entry_point(task),
        "stage": STAGE,
        "method": METHOD,
        "schema_version": OPT_SCHEMA_VERSION,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "optimized": True,
        "status": status,
        "optimization_status": optimization_status if status == "success" else "failed",
        "optimization_changed": optimization_changed,
        "optimized_success": status == "success" and optimized_success_for(str(optimization_status)),
        "fallback_used": status == "success" and optimization_status == "no_optimized_candidate",
        "issue": issue,
        "reason": reason,
        "solution": solution,
        "provider": provider,
        "model_name": model,
        "raw_provider": raw_provider,
        "raw_model_name": raw_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "created_at": utc_now(),
        "raw_contract_path": str(raw_contract_file) if raw_contract_file else None,
        "contract_type": "optimized_rl",
        "contract_hash": content_hash(contract) if isinstance(contract, dict) else None,
        "contract": contract if status == "success" else None,
        "candidate_contract": candidate_contract,
        "generation_prompt": prompt,
        "raw_response": raw_response,
        "api_result": api_result,
        "api_latency_seconds": api_result.get("latency_seconds") if isinstance(api_result, dict) else None,
        "error": error,
        "rl": rl,
        "source": task.get("source"),
        "source_version": task.get("source_version"),
        "selected_action": selected_action,
    }


def failed_record(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    raw_provider: str,
    raw_model: str,
    temperature: float,
    max_tokens: int,
    raw_contract_file: Path | None,
    error: str,
    issue: str,
    reason: str,
    solution: str,
) -> dict[str, Any]:
    return make_record(
        task=task,
        dataset=dataset,
        benchmark=benchmark,
        provider=provider,
        model=model,
        raw_provider=raw_provider,
        raw_model=raw_model,
        temperature=temperature,
        max_tokens=max_tokens,
        status="failed",
        raw_contract_file=raw_contract_file,
        contract=None,
        candidate_contract=None,
        prompt="",
        raw_response="",
        api_result=None,
        error=error,
        rl=None,
        issue=issue,
        reason=reason,
        solution=solution,
    )


def preserve_raw_passed_record(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    raw_provider: str,
    raw_model: str,
    raw_file: Path,
    raw_contract: dict[str, Any],
    state: dict[str, Any],
    features: dict[str, float],
    selection: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    diagnosis = diagnose_feedback({"available": True, "passed": True})
    rl_info = {
        "policy_path": str(policy_path(dataset, provider, model)),
        "selected_action": "keep_raw_if_passed",
        "action_description": ACTIONS["keep_raw_if_passed"],
        "state": state,
        "state_features": features,
        "selection": selection,
        "optimization_status": "preserved_raw_passed",
        "optimization_changed": False,
        "llm_called": False,
    }
    return make_record(
        task=task,
        dataset=dataset,
        benchmark=benchmark,
        provider=provider,
        model=model,
        raw_provider=raw_provider,
        raw_model=raw_model,
        temperature=temperature,
        max_tokens=max_tokens,
        status="success",
        raw_contract_file=raw_file,
        contract=raw_contract,
        candidate_contract=raw_contract,
        prompt="RL preserved raw/v1 contract because raw/v1 contract-guided evaluation passed.",
        raw_response="",
        api_result=None,
        error=None,
        rl=rl_info,
        **diagnosis,
    )


def no_optimized_candidate_record(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    raw_provider: str,
    raw_model: str,
    raw_file: Path,
    raw_contract: dict[str, Any],
    candidate: dict[str, Any] | None,
    state: dict[str, Any],
    features: dict[str, float],
    selected_action: str,
    selection: dict[str, Any],
    attempt: dict[str, Any] | None,
    feedback: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    diagnosis = diagnose_feedback(feedback)
    rl_info = {
        "policy_path": str(policy_path(dataset, provider, model)),
        "selected_action": selected_action,
        "action_description": ACTIONS[selected_action],
        "state": state,
        "state_features": features,
        "selection": selection,
        "optimization_status": "no_optimized_candidate",
        "optimization_changed": False,
        "llm_called": True,
        "attempt": attempt,
    }
    return make_record(
        task=task,
        dataset=dataset,
        benchmark=benchmark,
        provider=provider,
        model=model,
        raw_provider=raw_provider,
        raw_model=raw_model,
        temperature=temperature,
        max_tokens=max_tokens,
        status="success",
        raw_contract_file=raw_file,
        contract=raw_contract,
        candidate_contract=candidate or raw_contract,
        prompt="No meaningful optimized contract was produced in the single Stage 2 optimization attempt; preserving raw contract as non-optimized fallback.",
        raw_response="",
        api_result=None,
        error=None,
        rl=rl_info,
        issue="no_optimized_candidate",
        reason="Raw/v1 evaluation failed, but the single RL-selected optimization attempt did not produce a contract that changed the raw contract.",
        solution="Use the raw contract/code as fallback for this task in Stage 2, and analyze the missing semantic clause in reporting; do not perform per-task retry in this stage.",
    )


def generate_one(
    *,
    task: dict[str, Any],
    dataset: str,
    benchmark: str,
    provider: str,
    model: str,
    raw_provider: str,
    raw_model: str,
    client: Any,
    policy: dict[str, Any],
    raw_feedback: dict[str, Any],
    selection_mode: str,
    epsilon: float,
    softmax_temperature: float,
    rng: random.Random,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    tid = task_identifier(task)
    raw_file = raw_contract_path(dataset, tid, raw_provider, raw_model)
    raw_contract, raw_error = try_load_raw_contract(raw_file)

    if raw_feedback.get("available") is not True:
        diagnosis = diagnose_feedback(raw_feedback, raw_contract_error=raw_error)
        return failed_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            raw_provider=raw_provider,
            raw_model=raw_model,
            temperature=temperature,
            max_tokens=max_tokens,
            raw_contract_file=raw_file,
            error="Missing raw/v1 contract-guided evaluation feedback. Run raw evaluation before RL optimization.",
            **diagnosis,
        )

    if raw_contract is None:
        diagnosis = diagnose_feedback(raw_feedback, raw_contract_error=raw_error)
        return failed_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            raw_provider=raw_provider,
            raw_model=raw_model,
            temperature=temperature,
            max_tokens=max_tokens,
            raw_contract_file=raw_file,
            error=(
                "Invalid or missing raw/v1 contract. Strict Stage 2 RL does not regenerate or repair raw contracts; "
                "rerun v1 raw contract generation before optimized/RL contract generation."
            ),
            **diagnosis,
        )

    state = build_state(task, raw_contract, raw_feedback)
    features = features_from_state(state)
    selected_action, selection = select_action(
        policy=policy,
        features=features,
        raw_passed=raw_feedback.get("passed") is True,
        mode=selection_mode,
        epsilon=epsilon,
        temperature=softmax_temperature,
        rng=rng,
    )

    if selected_action == "keep_raw_if_passed":
        return preserve_raw_passed_record(
            task=task,
            dataset=dataset,
            benchmark=benchmark,
            provider=provider,
            model=model,
            raw_provider=raw_provider,
            raw_model=raw_model,
            raw_file=raw_file,
            raw_contract=raw_contract,
            state=state,
            features=features,
            selection=selection,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    prompt = build_revision_prompt(
        task=task,
        raw_contract=raw_contract,
        feedback=raw_feedback,
        selected_action=selected_action,
    )
    candidate: dict[str, Any] | None = None
    attempt_info: dict[str, Any] | None = None

    try:
        raw_response, api_result = call_chat_model(
            client=client,
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        parsed = parse_contract_response(raw_response)
        candidate = normalize_contract(parsed, task, dataset)
        optimization_status, optimization_changed = classify_candidate(raw_contract, candidate, raw_feedback)
        attempt_info = {
            "status": optimization_status,
            "changed": optimization_changed,
            "contract_hash": content_hash(candidate),
        }
        if optimization_changed:
            diagnosis = diagnose_feedback(raw_feedback)
            rl_info = {
                "policy_path": str(policy_path(dataset, provider, model)),
                "selected_action": selected_action,
                "action_description": ACTIONS[selected_action],
                "state": state,
                "state_features": features,
                "selection": selection,
                "optimization_status": "optimized",
                "optimization_changed": True,
                "llm_called": True,
                "attempt": attempt_info,
            }
            return make_record(
                task=task,
                dataset=dataset,
                benchmark=benchmark,
                provider=provider,
                model=model,
                raw_provider=raw_provider,
                raw_model=raw_model,
                temperature=temperature,
                max_tokens=max_tokens,
                status="success",
                raw_contract_file=raw_file,
                contract=candidate,
                candidate_contract=candidate,
                prompt=prompt,
                raw_response=raw_response,
                api_result=api_result,
                error=None,
                rl=rl_info,
                **diagnosis,
            )
    except Exception as exc:
        attempt_info = {"status": "error", "error": short(exc, 500)}
        logger.exception("RL optimized contract generation failed for %s", tid)

    return no_optimized_candidate_record(
        task=task,
        dataset=dataset,
        benchmark=benchmark,
        provider=provider,
        model=model,
        raw_provider=raw_provider,
        raw_model=raw_model,
        raw_file=raw_file,
        raw_contract=raw_contract,
        candidate=candidate,
        state=state,
        features=features,
        selected_action=selected_action,
        selection=selection,
        attempt=attempt_info,
        feedback=raw_feedback,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def resolve_models(args: argparse.Namespace) -> None:
    args.model = args.model or default_model(args.provider)
    args.raw_provider = args.raw_provider or args.provider
    args.raw_model = args.raw_model or (args.model if args.raw_provider == args.provider else default_model(args.raw_provider))
    args.feedback_provider = args.feedback_provider or args.raw_provider
    args.feedback_model = args.feedback_model or args.raw_model


def run(args: argparse.Namespace) -> None:
    resolve_models(args)
    dataset_info = DATASETS[args.dataset]
    benchmark = dataset_info["label"]
    client = get_client(args.provider)
    rng = random.Random(args.seed)
    tasks = select_tasks(load_json_list(dataset_info["path"]), start=args.start, count=args.count)
    raw_feedback_by_task = evaluation_map(
    feedback_results_folder(
        "raw_contracts",
        args.dataset,
        args.feedback_provider,
        args.feedback_model,
    )
)
    ppath = policy_path(args.dataset, args.provider, args.model)
    policy = load_policy(ppath)

    updates = 0
    if not args.no_policy_update:
        updates = update_policy_from_optimized_results(
            policy=policy,
            dataset=args.dataset,
            provider=args.provider,
            model=args.model,
            raw_feedback_by_task=raw_feedback_by_task,
            learning_rate=args.learning_rate,
            selection_temperature=args.softmax_temperature,
        )
        save_json(ppath, policy)

    counts = {"optimized": 0, "preserved": 0, "no_candidate": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        tid = task_identifier(task)
        out = optimized_contract_path(args.dataset, tid, args.provider, args.model)
        if out.exists() and not args.overwrite:
            counts["skipped"] += 1
            print(f"[{index}] SKIP {tid}")
            continue

        record = generate_one(
            task=task,
            dataset=args.dataset,
            benchmark=benchmark,
            provider=args.provider,
            model=args.model,
            raw_provider=args.raw_provider,
            raw_model=args.raw_model,
            client=client,
            policy=policy,
            raw_feedback=make_feedback(raw_feedback_by_task.get(tid)),
            selection_mode=args.selection,
            epsilon=args.epsilon,
            softmax_temperature=args.softmax_temperature,
            rng=rng,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        save_json(out, record)

        status = record.get("optimization_status")
        if record["status"] != "success":
            counts["failed"] += 1
            print(f"[{index}] FAIL {tid}: {record['error']}")
        elif status == "preserved_raw_passed":
            counts["preserved"] += 1
            print(f"[{index}] KEEP {tid} — raw passed; contract preserved")
        elif status == "no_optimized_candidate":
            counts["no_candidate"] += 1
            print(f"[{index}] NO-CANDIDATE {tid} — raw failed; preserving raw as non-optimized fallback")
        else:
            counts["optimized"] += 1
            print(f"[{index}] OPTIMIZED {tid}")

        if args.delay > 0:
            time.sleep(args.delay)

    print("\nStage 2D RL optimized contract generation finished")
    print(f"Dataset: {args.dataset} ({benchmark})")
    print(f"Provider/model: {args.provider}/{args.model}")
    print(f"Raw contract source: {args.raw_provider}/{args.raw_model}")
    print(f"Raw evaluation feedback source: {args.feedback_provider}/{args.feedback_model}")
    print(f"Policy: {ppath}")
    print(f"Policy updates from previous optimized evaluations: {updates}")
    print(json.dumps(counts, indent=2))
    print(f"Elapsed: {round(time.perf_counter() - started, 4)}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2D: monotonic RL optimized contract generation")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--raw-provider", choices=sorted(PROVIDERS))
    parser.add_argument("--raw-model")
    parser.add_argument("--feedback-provider", choices=sorted(PROVIDERS))
    parser.add_argument("--feedback-model")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--selection", choices=("epsilon_greedy", "softmax", "greedy", "random"), default="epsilon_greedy")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--softmax-temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-policy-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging(LOG_FILE)
    run(parse_args())


if __name__ == "__main__":
    main()
