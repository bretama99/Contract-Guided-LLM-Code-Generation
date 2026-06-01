from __future__ import annotations

from pathlib import Path

from src.classeval.core import JsonDict, task_id, text
from src.classeval.execution import FAILURE_OTHER, normalize_failure_type
from src.common.io_utils import load_json

OPTIMIZED_SUCCESS_STATUS = "success"


def executed_test_metrics(evaluation: JsonDict) -> JsonDict:
    metrics = ((evaluation.get("evaluation") or {}).get("metrics") or {})
    return metrics if isinstance(metrics, dict) else {}


def reward_from_evaluation(evaluation: JsonDict | None) -> tuple[float, JsonDict]:
    if not isinstance(evaluation, dict):
        return 0.0, {
            "status": "missing_evaluation",
            "failure_type": FAILURE_OTHER,
            "executed_tests": False,
            "tests_passed": None,
            "tests_total": None,
            "error": None,
        }

    metrics = executed_test_metrics(evaluation)
    total = metrics.get("total")
    passed = metrics.get("passed")
    executed = isinstance(total, int) and total > 0

    if evaluation.get("passed") is True and executed:
        reward = 1.0
    elif executed and isinstance(passed, int):
        reward = max(0.0, min(1.0, passed / total))
    else:
        reward = 0.0

    return reward, {
        "status": text(evaluation.get("status")) or ("passed" if evaluation.get("passed") else "failed"),
        "failure_type": None if evaluation.get("passed") else normalize_failure_type(evaluation.get("failure_type")),
        "executed_tests": executed,
        "tests_passed": passed,
        "tests_total": total,
        "error": evaluation.get("error"),
    }


def load_evaluation_reward(path: Path | None) -> tuple[float, JsonDict]:
    if path is None or not path.exists():
        return reward_from_evaluation(None)
    return reward_from_evaluation(load_json(path))


def contract_record_status(record: JsonDict | None) -> JsonDict:
    if not isinstance(record, dict):
        return {
            "status": "missing",
            "failure_type": FAILURE_OTHER,
            "fallback": False,
            "usable_for_optimized_metrics": False,
            "contract_type": None,
            "contract_version": None,
            "error": None,
        }

    status = text(record.get("status")) or "unknown"
    fallback = record.get("fallback_to_previous_contract") is True or status == "fallback"
    usable = status == OPTIMIZED_SUCCESS_STATUS and not fallback and isinstance(record.get("contract"), dict)

    return {
        "status": status,
        "failure_type": None if status == OPTIMIZED_SUCCESS_STATUS else normalize_failure_type(record.get("failure_type")),
        "fallback": fallback,
        "usable_for_optimized_metrics": usable,
        "contract_type": record.get("contract_type"),
        "contract_version": record.get("contract_version"),
        "error": record.get("error"),
    }


def reward_log_entry(
    task: JsonDict,
    *,
    provider: str,
    model: str,
    contract_version: str,
    contract_path: Path | None,
    contract_status: JsonDict,
    evaluation_path: Path | None,
    reward: float,
    evaluation_result: JsonDict,
) -> JsonDict:
    return {
        "task_id": task_id(task),
        "provider": provider,
        "model_name": model,
        "contract_version": contract_version,
        "contract_path": None if contract_path is None else str(contract_path),
        "contract_status": contract_status.get("status"),
        "contract_type": contract_status.get("contract_type"),
        "fallback": contract_status.get("fallback") is True,
        "usable_for_optimized_metrics": contract_status.get("usable_for_optimized_metrics") is True,
        "status": evaluation_result.get("status"),
        "failure_type": evaluation_result.get("failure_type"),
        "reward": round(float(reward), 6),
        "evaluation_path": None if evaluation_path is None else str(evaluation_path),
        "evaluation_result": evaluation_result,
    }


__all__ = [
    "OPTIMIZED_SUCCESS_STATUS",
    "contract_record_status",
    "executed_test_metrics",
    "load_evaluation_reward",
    "reward_from_evaluation",
    "reward_log_entry",
]