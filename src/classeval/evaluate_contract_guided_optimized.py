from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.classeval.core import JsonDict, as_dict, as_list, task_id, text
from src.classeval.evaluate_contract_guided import evaluation_path as v1_evaluation_path
from src.classeval.execution import evaluate_generation, failure_record, normalize_failure_type
from src.classeval.generate_contracts import TASK_FILE
from src.classeval.generate_from_optimized_contracts import generation_path
from src.classeval.rl_generate_optimized_contracts import (
    ACTIONS,
    ACTION_IDS,
    FEATURE_NAMES,
    default_policy_path,
    optimized_path,
)
from src.classeval.rl_reward import contract_record_status, reward_from_evaluation, reward_log_entry
from src.common.config import LOG_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model
from src.common.task_utils import select_tasks

STAGE = "2R_feedback_optimized_contract_guided_evaluation"
EVAL_DIR = OUTPUT_ROOT / "classeval" / "evaluation" / "rl_optimized_contract_guided"
RESULT_DIR = RESULTS_ROOT / "classeval" / "rl_optimized_contract_guided"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contract_guided_eval.log"


def evaluation_path(
    task: JsonDict,
    provider: str,
    model: str,
    contract_provider: str | None = None,
    contract_model: str | None = None,
) -> Path:
    base = EVAL_DIR / safe_name(provider) / safe_name(model)
    if contract_provider and contract_model and (contract_provider != provider or contract_model != model):
        base = base / "from_contract" / safe_name(contract_provider) / safe_name(contract_model)
    return base / f"{safe_name(task_id(task))}.json"


def compact_text(value: Any, limit: int = 4000) -> str:
    value = text(value)
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def classify_error(kind: Any, detail: Any = "") -> str:
    raw = f"{text(kind)} {text(detail)}".lower()

    if "missing method" in raw or "incomplete method" in raw:
        return "missing method"
    if "signature" in raw or "staticmethod" in raw:
        return "signature error"
    if "syntax" in raw or "invalid python" in raw or "indentation" in raw:
        return "syntax error"
    if "import" in raw or "no module named" in raw or "modulenotfounderror" in raw:
        return "import error"
    if "timeout" in raw or "timed out" in raw:
        return "timeout"

    return normalize_failure_type(kind or detail)


def generation_status(generation: JsonDict | None) -> JsonDict:
    generation = generation if isinstance(generation, dict) else {}
    return {
        "generation_status": generation.get("status"),
        "generation_failure_type": classify_error(generation.get("failure_type"), generation.get("failure_detail") or generation.get("error")),
        "generation_failure_detail": compact_text(generation.get("failure_detail") or generation.get("error")),
        "copied_from_v1": generation.get("copied_from_v1") is True,
        "contract_path": generation.get("contract_path"),
        "contract_status": generation.get("contract_status"),
        "contract_type": generation.get("contract_type"),
        "contract_version": generation.get("contract_version"),
        "contract_provider": generation.get("contract_provider"),
        "contract_model": generation.get("contract_model"),
    }


def feedback_for_next_contract(result: JsonDict) -> JsonDict:
    metrics = ((result.get("evaluation") or {}).get("metrics") or {})
    feedback: JsonDict = {
        "passed": result.get("passed") is True,
        "failure_type": result.get("failure_type"),
    }

    for key, target in (
        ("total", "tests_total"),
        ("passed", "tests_passed"),
        ("failed", "tests_failed"),
        ("failures", "failures"),
        ("errors", "errors"),
        ("skipped", "skipped"),
    ):
        if metrics.get(key) is not None:
            feedback[target] = metrics.get(key)

    if result.get("passed") is not True:
        feedback["evaluation_error"] = compact_text(result.get("failure_detail") or result.get("error"))

    return {key: value for key, value in feedback.items() if value not in (None, "", [], {})}


def load_v1_result(task: JsonDict, provider: str, model: str) -> tuple[JsonDict | None, Path]:
    path = v1_evaluation_path(task, provider, model)
    return (load_json(path) if path.exists() else None), path


def reward_value(result: JsonDict | None) -> float:
    reward, _ = reward_from_evaluation(result)
    return float(reward)


def attach_reward_log(
    task: JsonDict,
    provider: str,
    model: str,
    result: JsonDict,
    generation: JsonDict | None,
    contract_provider: str,
    contract_model: str,
) -> JsonDict:
    generation = generation if isinstance(generation, dict) else {}

    raw_path = generation.get("contract_path")
    contract_path = Path(raw_path) if raw_path else optimized_path(task, contract_provider, contract_model)
    contract_record = load_json(contract_path) if contract_path.exists() else None
    contract_status = contract_record_status(contract_record)

    reward, evaluation_result = reward_from_evaluation(result)

    result["reward"] = round(reward, 6)
    result["rl_reward_log"] = reward_log_entry(
        task,
        provider=provider,
        model=model,
        contract_version=contract_status.get("contract_version") or generation.get("contract_version") or "v2_rl_optimized",
        contract_path=contract_path,
        contract_status=contract_status,
        evaluation_path=evaluation_path(task, provider, model, contract_provider, contract_model),
        reward=reward,
        evaluation_result=evaluation_result,
    )

    return result


def add_safe_selection(result: JsonDict, task: JsonDict, base_provider: str, base_model: str) -> JsonDict:
    v1, v1_path = load_v1_result(task, base_provider, base_model)

    opt_reward = reward_value(result)
    v1_reward = reward_value(v1)

    opt_passed = result.get("passed") is True
    v1_passed = isinstance(v1, dict) and v1.get("passed") is True
    opt_usable = (result.get("rl_reward_log") or {}).get("usable_for_optimized_metrics") is True

    strict_success = bool(opt_passed and opt_usable and opt_reward > v1_reward)

    if strict_success:
        source = "optimized"
        reason = "strict_optimized_improvement"
    elif v1_passed:
        source = "v1"
        reason = "v1_passed_or_optimized_not_better"
    elif opt_usable and opt_reward > v1_reward:
        source = "optimized"
        reason = "optimized_partial_improvement"
    else:
        source = "v1"
        reason = "tie_or_regression_no_optimized_success"

    if source == "optimized":
        selected_passed = opt_passed
        selected_reward = opt_reward
        selected_failure_type = None if opt_passed else classify_error(result.get("failure_type"), result.get("failure_detail") or result.get("error"))
        selected_detail = result.get("failure_detail") or result.get("error")
    else:
        selected_passed = v1_passed
        selected_reward = v1_reward
        selected_failure_type = None if v1_passed else classify_error(
            v1.get("failure_type") if isinstance(v1, dict) else "v1_missing_evaluation",
            (v1.get("failure_detail") or v1.get("error")) if isinstance(v1, dict) else str(v1_path),
        )
        selected_detail = None if v1_passed else (
            (v1.get("failure_detail") or v1.get("error")) if isinstance(v1, dict) else str(v1_path)
        )

    result["safe_selection"] = {
        "selected_source": source,
        "selection_reason": reason,
        "selected_passed": selected_passed,
        "selected_reward": round(float(selected_reward), 6),
        "selected_failure_type": selected_failure_type,
        "selected_failure_detail": compact_text(selected_detail),
        "optimized_passed": opt_passed,
        "optimized_reward": round(float(opt_reward), 6),
        "optimized_usable_for_metrics": opt_usable,
        "v1_passed": v1_passed,
        "v1_reward": round(float(v1_reward), 6),
        "v1_evaluation_path": str(v1_path),
        "optimized_improved_over_v1": bool(opt_usable and opt_reward > v1_reward),
        "optimized_tied_v1": bool(opt_reward == v1_reward),
        "optimized_regressed_from_v1": bool(opt_reward < v1_reward),
        "strict_optimized_success": strict_success,
        "strict_optimized_success_rule": (
            "optimized must pass, be usable, and improve over v1 reward; "
            "ties, fallbacks, failed generations, unchanged contracts, and regressions are not optimized success"
        ),
        "used_no_regression_gate": True,
    }

    return result


def evaluate_one(
    task: JsonDict,
    provider: str,
    model: str,
    timeout: float,
    include_code: bool,
    contract_provider: str,
    contract_model: str,
    base_provider: str,
    base_model: str,
) -> JsonDict:
    gen_path = generation_path(task, provider, model, contract_provider, contract_model)

    if not gen_path.exists():
        detail = f"Missing generation file: {gen_path}"
        result = failure_record(
            task,
            provider=provider,
            model=model,
            stage=STAGE,
            failure_type="missing method",
            error=detail,
            generation_path=gen_path,
            failure_stage="generation",
        )
        result["failure_detail"] = detail
        result["feedback_for_next_contract"] = feedback_for_next_contract(result)
        result = attach_reward_log(task, provider, model, result, None, contract_provider, contract_model)
        return add_safe_selection(result, task, base_provider, base_model)

    generation = load_json(gen_path)

    if generation.get("status") != "success":
        detail = generation.get("failure_detail") or generation.get("error") or "Optimized generation failed."
        failure_type = classify_error(generation.get("failure_type"), detail)
        result = failure_record(
            task,
            provider=provider,
            model=model,
            stage=STAGE,
            failure_type=failure_type,
            error=detail,
            generation_path=gen_path,
            failure_stage="generation",
        )
        result["failure_detail"] = compact_text(detail)
        result["generation"] = {"path": str(gen_path), **generation_status(generation)}
        result["feedback_for_next_contract"] = feedback_for_next_contract(result)
        result = attach_reward_log(task, provider, model, result, generation, contract_provider, contract_model)
        return add_safe_selection(result, task, base_provider, base_model)

    result = evaluate_generation(
        task,
        generation,
        gen_path,
        provider=provider,
        model=model,
        stage=STAGE,
        timeout=timeout,
        include_code=include_code,
    )

    if result.get("passed") is not True:
        detail = result.get("failure_detail") or result.get("error")
        result["failure_type"] = classify_error(result.get("failure_type"), detail)
        result["failure_detail"] = compact_text(detail)
    else:
        result["failure_detail"] = None

    result["generation"] = {"path": str(gen_path), **generation_status(generation)}
    result["feedback_for_next_contract"] = feedback_for_next_contract(result)
    result = attach_reward_log(task, provider, model, result, generation, contract_provider, contract_model)
    return add_safe_selection(result, task, base_provider, base_model)


def result_name(provider: str, model: str, contract_provider: str, contract_model: str) -> str:
    name = f"classeval_feedback_optimized_{safe_name(provider)}_{safe_name(model)}"
    if contract_provider != provider or contract_model != model:
        name += f"_from_contract_{safe_name(contract_provider)}_{safe_name(contract_model)}"
    return name


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json_or_none(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    value = load_json(path)
    return value if isinstance(value, dict) else None


def contract_changed(record: JsonDict | None) -> bool:
    if not isinstance(record, dict) or not isinstance(record.get("contract"), dict):
        return False

    previous_file = text(record.get("previous_contract_file"))
    if not previous_file:
        return True

    previous = load_json_or_none(Path(previous_file))
    previous_contract = previous.get("contract") if isinstance(previous, dict) else None

    if not isinstance(previous_contract, dict):
        return True

    return canonical(record["contract"]) != canonical(previous_contract)


def policy_path(provider: str, model: str, explicit: str | None = None) -> Path:
    return Path(explicit) if explicit else default_policy_path(provider, model)


def fresh_policy() -> JsonDict:
    width = len(FEATURE_NAMES)
    return {
        "stage": "2R_contextual_bandit_policy",
        "policy_type": "contextual_bandit_softmax",
        "policy_version": "contextual_bandit_in_3_files_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": None,
        "feature_names": FEATURE_NAMES,
        "actions": ACTIONS,
        "weights": {action_id: [0.0] * width for action_id in ACTION_IDS},
        "counts": {action_id: 0 for action_id in ACTION_IDS},
        "total_updates": 0,
        "reward_baseline": 0.0,
        "learning_rate": 0.08,
        "temperature": 1.0,
        "epsilon": 0.15,
        "l2": 0.0005,
        "optimized_policy_text": (
            "Contextual-bandit RL policy. Select one contract-revision action from the current state. "
            "Reward is positive only when optimized output truly improves over v1."
        ),
        "last_update_summary": {},
        "recent_updates": [],
    }


def load_policy(path: Path) -> JsonDict:
    policy = fresh_policy()

    if path.exists():
        data = load_json(path)
        if isinstance(data, dict):
            policy.update(data)

    width = len(FEATURE_NAMES)
    weights = as_dict(policy.get("weights"))
    policy["weights"] = {
        action_id: ([float(x) for x in as_list(weights.get(action_id))[:width]] + [0.0] * width)[:width]
        for action_id in ACTION_IDS
    }

    counts = as_dict(policy.get("counts"))
    policy["counts"] = {action_id: int(counts.get(action_id) or 0) for action_id in ACTION_IDS}
    policy["feature_names"] = FEATURE_NAMES
    policy["actions"] = ACTIONS

    return policy


def vector(features: JsonDict) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    temperature = max(1e-6, float(temperature))
    top = max(scores.values()) if scores else 0.0
    exps = {key: math.exp((value - top) / temperature) for key, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def selected_action(contract_record: JsonDict | None) -> str:
    policy = as_dict((contract_record or {}).get("rl_policy"))
    action = as_dict(policy.get("selected_action"))
    return text(policy.get("action_id")) or text(action.get("id"))


def state_features(contract_record: JsonDict | None) -> JsonDict:
    return as_dict(as_dict((contract_record or {}).get("rl_policy")).get("state_features"))


def strict_learning_reward(
    task: JsonDict,
    provider: str,
    model: str,
    contract_provider: str,
    contract_model: str,
    base_provider: str,
    base_model: str,
) -> JsonDict | None:
    contract_path = optimized_path(task, contract_provider, contract_model)
    contract_record = load_json_or_none(contract_path)

    if contract_record is None:
        return None

    action_id = selected_action(contract_record)
    features = state_features(contract_record)

    if not action_id or not features:
        return None

    opt_eval_path = evaluation_path(task, provider, model, contract_provider, contract_model)
    v1_eval_path = v1_evaluation_path(task, base_provider, base_model)

    opt_eval = load_json_or_none(opt_eval_path)
    v1_eval = load_json_or_none(v1_eval_path)

    opt_reward, opt_info = reward_from_evaluation(opt_eval)
    v1_reward, v1_info = reward_from_evaluation(v1_eval)

    status = contract_record_status(contract_record)

    opt_passed = isinstance(opt_eval, dict) and opt_eval.get("passed") is True
    usable = status.get("usable_for_optimized_metrics") is True
    fallback = status.get("fallback") is True
    changed = contract_changed(contract_record)

    strict = bool(opt_passed and usable and not fallback and changed and opt_reward > v1_reward)

    reason = "improved"
    if fallback:
        reason = "fallback_contract"
    elif not usable:
        reason = "contract_not_usable"
    elif not changed:
        reason = "unchanged_contract"
    elif not opt_passed:
        reason = "optimized_not_passed"
    elif opt_reward <= v1_reward:
        reason = "not_better_than_v1"

    return {
        "task_id": task_id(task),
        "selected_action": action_id,
        "state_features": features,
        "reward": round(float(opt_reward), 6) if strict else 0.0,
        "strict_optimized_success": strict,
        "strict_optimized_reason": reason,
        "optimized_score": round(float(opt_reward), 6),
        "v1_score": round(float(v1_reward), 6),
        "optimized_result": opt_info,
        "v1_result": v1_info,
        "contract_changed": changed,
        "contract_usable": usable,
        "fallback": fallback,
        "contract_path": str(contract_path),
        "optimized_evaluation_path": str(opt_eval_path),
        "v1_evaluation_path": str(v1_eval_path),
    }


def update_policy(policy: JsonDict, features: JsonDict, action_id: str, reward: float) -> JsonDict:
    if action_id not in ACTION_IDS:
        raise ValueError(f"Unknown RL action: {action_id}")

    reward = max(0.0, min(1.0, float(reward)))
    x = vector(features)

    weights = as_dict(policy.get("weights"))
    scores = {
        action: dot([float(value) for value in as_list(weights.get(action))], x)
        for action in ACTION_IDS
    }
    probabilities = softmax(scores, float(policy.get("temperature", 1.0)))

    old_baseline = float(policy.get("reward_baseline") or 0.0)
    total_updates = int(policy.get("total_updates") or 0)
    advantage = reward - old_baseline
    lr = float(policy.get("learning_rate") or 0.08)
    l2 = float(policy.get("l2") or 0.0005)

    for action in ACTION_IDS:
        grad = (1.0 if action == action_id else 0.0) - probabilities[action]
        current = [float(value) for value in as_list(weights.get(action))]
        policy["weights"][action] = [
            (1.0 - lr * l2) * weight + lr * advantage * grad * value
            for weight, value in zip(current, x)
        ]

    policy["counts"][action_id] = int(policy["counts"].get(action_id) or 0) + 1
    policy["total_updates"] = total_updates + 1

    alpha = min(0.2, 1.0 / max(1, policy["total_updates"]))
    policy["reward_baseline"] = (1.0 - alpha) * old_baseline + alpha * reward
    policy["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return {
        "action_id": action_id,
        "reward": round(reward, 6),
        "advantage": round(advantage, 6),
        "action_probability": round(probabilities[action_id], 6),
        "old_baseline": round(old_baseline, 6),
        "new_baseline": round(float(policy["reward_baseline"]), 6),
        "total_updates": policy["total_updates"],
    }


def update_rl_policy_from_results(
    tasks: list[JsonDict],
    provider: str,
    model: str,
    contract_provider: str,
    contract_model: str,
    base_provider: str,
    base_model: str,
    explicit_policy_file: str | None = None,
) -> JsonDict:
    path = policy_path(contract_provider, contract_model, explicit_policy_file)
    policy = load_policy(path)

    updated = 0
    skipped = 0
    strict_successes = 0
    reasons = Counter()
    rewards: list[float] = []
    logs: list[JsonDict] = []

    for task in tasks:
        record = strict_learning_reward(
            task,
            provider,
            model,
            contract_provider,
            contract_model,
            base_provider,
            base_model,
        )

        if record is None:
            skipped += 1
            continue

        update = update_policy(policy, record["state_features"], record["selected_action"], float(record["reward"]))
        record["policy_update"] = update

        updated += 1
        strict_successes += int(record["strict_optimized_success"] is True)
        reasons[record["strict_optimized_reason"]] += 1
        rewards.append(float(record["reward"]))
        logs.append(record)

    summary = {
        "policy_file": str(path),
        "updated": updated,
        "skipped": skipped,
        "strict_optimized_successes": strict_successes,
        "strict_optimized_success_rate": strict_successes / updated if updated else 0.0,
        "mean_learning_reward": sum(rewards) / updated if updated else 0.0,
        "strict_failure_reasons": dict(reasons),
        "reward_rule": (
            "positive only when optimized passes, optimized contract is usable, non-fallback, changed, "
            "and optimized reward is greater than v1 reward"
        ),
    }

    policy["last_update_summary"] = summary
    policy["recent_updates"] = logs[-100:]
    save_json(path, policy)

    return summary


def final_summary(results: list[JsonDict], timeout: float) -> JsonDict:
    total = len(results)

    selected_passed = sum((r.get("safe_selection") or {}).get("selected_passed") is True for r in results)
    raw_optimized_passed = sum((r.get("safe_selection") or {}).get("optimized_passed") is True for r in results)
    strict_success = sum((r.get("safe_selection") or {}).get("strict_optimized_success") is True for r in results)
    improved = sum((r.get("safe_selection") or {}).get("optimized_improved_over_v1") is True for r in results)
    tied = sum((r.get("safe_selection") or {}).get("optimized_tied_v1") is True for r in results)
    regressed = sum((r.get("safe_selection") or {}).get("optimized_regressed_from_v1") is True for r in results)

    selected_failures = Counter()
    optimized_failures = Counter()

    for result in results:
        selection = result.get("safe_selection") or {}

        if selection.get("selected_passed") is not True:
            selected_failures[selection.get("selected_failure_type") or "evaluation error"] += 1

        if selection.get("optimized_passed") is not True:
            optimized_failures[classify_error(result.get("failure_type"), result.get("failure_detail") or result.get("error"))] += 1

    return {
        "selected_no_regression_passed": selected_passed,
        "selected_no_regression_failed": total - selected_passed,
        "selected_no_regression_pass@1": selected_passed / total if total else 0.0,
        "selected_no_regression_pass@1_percent": 100.0 * selected_passed / total if total else 0.0,
        "raw_optimized_passed": raw_optimized_passed,
        "raw_optimized_pass@1": raw_optimized_passed / total if total else 0.0,
        "raw_optimized_pass@1_percent": 100.0 * raw_optimized_passed / total if total else 0.0,
        "strict_optimized_successes": strict_success,
        "strict_optimized_success_rate": strict_success / total if total else 0.0,
        "strict_optimized_success_percent": 100.0 * strict_success / total if total else 0.0,
        "optimized_improved_over_v1": improved,
        "optimized_tied_v1": tied,
        "optimized_regressed_from_v1": regressed,
        "strict_optimized_success_rule": (
            "optimized must pass, be usable, and improve over v1 reward; "
            "selected v1 fallbacks, ties, unchanged contracts, failed generations, and regressions are not counted"
        ),
        "timeout_seconds": timeout,
        "selected_failure_counts": dict(selected_failures),
        "optimized_failure_counts": dict(optimized_failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optimized ClassEval outputs and update embedded contextual-bandit RL policy.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--contract-provider")
    parser.add_argument("--contract-model")
    parser.add_argument("--base-provider")
    parser.add_argument("--base-model")
    parser.add_argument("--policy-file")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument("--no-policy-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    contract_provider = args.contract_provider or args.provider
    contract_model = args.contract_model or model
    base_provider = args.base_provider or args.provider
    base_model = args.base_model or model

    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    results: list[JsonDict] = []
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = evaluation_path(task, args.provider, model, contract_provider, contract_model)

        if path.exists() and not args.overwrite:
            result = load_json(path)
            result = add_safe_selection(result, task, base_provider, base_model)
        else:
            result = evaluate_one(
                task,
                args.provider,
                model,
                args.timeout,
                args.include_code,
                contract_provider,
                contract_model,
                base_provider,
                base_model,
            )
            save_json(path, result)

        result["index"] = index
        results.append(result)

        selected = result.get("safe_selection") or {}
        status = "PASS" if selected.get("selected_passed") is True else f"FAIL/{selected.get('selected_failure_type')}"
        print(
            f"[{index}] {status}: {task_id(task)} "
            f"selected={selected.get('selected_source')} "
            f"reason={selected.get('selection_reason')}"
        )

    summary = final_summary(results, args.timeout)

    if not args.no_policy_update:
        summary["rl_policy_update"] = update_rl_policy_from_results(
            tasks,
            args.provider,
            model,
            contract_provider,
            contract_model,
            base_provider,
            base_model,
            args.policy_file,
        )

    summary["elapsed_seconds"] = round(time.perf_counter() - started, 4)

    name = result_name(args.provider, model, contract_provider, contract_model)
    save_json(RESULT_DIR / f"{name}_summary.json", summary)
    save_json(RESULT_DIR / f"{name}_details.json", results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()