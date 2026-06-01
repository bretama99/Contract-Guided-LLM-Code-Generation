from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.contract_normalizer import normalize_contract_shape, stage2_shape_errors
from src.classeval.core import JsonDict, as_dict, as_list, ensure_no_reference_leak, skeleton, task_id, text
from src.classeval.evaluate_contract_guided import evaluation_path as v1_evaluation_path
from src.classeval.generate_contracts import (
    DEFAULT_MAX_TOKENS,
    SYSTEM_PROMPT,
    TASK_FILE,
    contract_path as v1_contract_path,
)
from src.classeval.generate_from_contracts import generation_path as v1_generation_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, call_chat_model, default_model, get_client
from src.common.task_utils import select_tasks

STAGE = "2R_feedback_optimized_contract_generation"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_optimized_contracts"
POLICY_DIR = OUTPUT_ROOT / "classeval" / "rl_revision_policy"
LOG_FILE = LOG_ROOT / "classeval_rl_optimized_contracts.log"

logger = logging.getLogger(__name__)

ACTIONS: list[JsonDict] = [
    {
        "id": "strengthen_preconditions",
        "name": "Strengthen preconditions",
        "instruction": "Clarify supported input and state requirements without inventing new exceptions.",
    },
    {
        "id": "strengthen_postconditions",
        "name": "Strengthen postconditions",
        "instruction": "Clarify outputs, return values, state mutations, ordering, formatting, and side effects.",
    },
    {
        "id": "add_invariants",
        "name": "Add invariants",
        "instruction": "Add concise persistent class-state rules only when the class has state.",
    },
    {
        "id": "add_edge_cases",
        "name": "Add edge cases",
        "instruction": "Add visible edge cases such as empty input, boundaries, duplicates, repeated calls, and missing keys.",
    },
    {
        "id": "fix_invalid_input_behavior",
        "name": "Fix invalid input behavior",
        "instruction": "Clarify invalid-input behavior only when the visible task supports it.",
    },
    {
        "id": "simplify_vague_contracts",
        "name": "Simplify vague contracts",
        "instruction": "Remove vague, redundant, contradictory, or over-strict clauses.",
    },
    {
        "id": "failure_guided_repair",
        "name": "Failure-guided repair",
        "instruction": "Use previous failure type and pass ratio to repair the most likely contract weakness.",
    },
]

ACTION_IDS = [str(action["id"]) for action in ACTIONS]

FEATURE_NAMES = [
    "bias",
    "method_count",
    "field_count",
    "contract_coverage",
    "avg_postconditions",
    "avg_edge_cases",
    "class_invariant_count",
    "vague_ratio",
    "prev_passed",
    "pass_ratio",
    "failed_generation",
    "failed_logical",
    "failed_runtime",
    "failed_syntax",
    "failed_timeout",
    "failed_other",
]


def optimized_path(task: JsonDict, provider: str, model: str) -> Path:
    return OUT_DIR / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def optimized_generation_path(task: JsonDict, provider: str, model: str) -> Path:
    return OUTPUT_ROOT / "classeval" / "rl_optimized_contract_guided" / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def optimized_evaluation_path(task: JsonDict, provider: str, model: str) -> Path:
    return OUTPUT_ROOT / "classeval" / "evaluation" / "rl_optimized_contract_guided" / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}.json"


def default_policy_path(provider: str, model: str) -> Path:
    return POLICY_DIR / safe_name(provider) / safe_name(model) / "policy.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_class_name(task: JsonDict) -> str:
    return text(task.get("class_name")) or text(task.get("entry_point")) or "UnknownClass"


def classify_contract_error(error: Any) -> tuple[str, str]:
    detail = str(error).strip()
    lower = detail.lower()

    if not detail:
        return "contract_generation_unknown_error", ""
    if "no valid json object found" in lower:
        return "contract_invalid_json", detail
    if "unterminated string" in lower:
        return "contract_invalid_json_truncated", detail
    if "stage 2 shape errors" in lower:
        return "contract_shape_error", detail
    if "policy file not found" in lower:
        return "policy_missing", detail
    if "rate limit" in lower:
        return "provider_rate_limit", detail
    if "authentication" in lower or "api key" in lower:
        return "provider_authentication_error", detail
    if "connection" in lower or "network" in lower:
        return "provider_connection_error", detail
    if "timed out" in lower or "timeout" in lower:
        return "timeout", detail

    return "contract_generation_failed", detail


def method_names(task: JsonDict) -> list[str]:
    names = [text(name) for name in as_list(task.get("method_names")) if text(name)]
    if names:
        return names

    return [
        text(info.get("method_name"))
        for info in as_list(task.get("methods_info"))
        if isinstance(info, dict) and text(info.get("method_name"))
    ]


def safe_methods_info(task: JsonDict) -> list[JsonDict]:
    blocked = (
        "test",
        "tests",
        "unit_test",
        "solution",
        "answer",
        "reference",
        "expected",
        "oracle",
        "ground_truth",
        "assert",
        "stdout",
        "stderr",
        "traceback",
    )

    cleaned: list[JsonDict] = []
    for info in as_list(task.get("methods_info")):
        if isinstance(info, dict):
            cleaned.append({str(k): v for k, v in info.items() if not any(t in str(k).lower() for t in blocked)})

    return cleaned


def task_view(task: JsonDict) -> JsonDict:
    imports = task.get("import_statement")
    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
        "entry_point": text(task.get("entry_point")) or safe_class_name(task),
        "skeleton": text(task.get("skeleton")),
        "class_description": text(task.get("class_description")),
        "class_constructor": text(task.get("class_constructor")),
        "fields": as_list(task.get("fields")),
        "imports": imports if isinstance(imports, list) else [],
        "method_names": method_names(task),
        "methods_info": safe_methods_info(task),
    }


def method_signature(info: JsonDict) -> str:
    for line in text(info.get("method_description")).splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line
    return ""


def constructor_signature(task: JsonDict) -> str:
    for line in text(task.get("class_constructor")).splitlines():
        line = line.strip()
        if line.startswith("def __init__"):
            return line
    return ""


def method_is_static(task: JsonDict, name: str) -> bool:
    pattern = r"@staticmethod\s*\n\s*def\s+" + re.escape(name) + r"\s*\("
    if re.search(pattern, text(task.get("skeleton"))):
        return True

    for info in as_list(task.get("methods_info")):
        if isinstance(info, dict) and text(info.get("method_name")) == name:
            return bool(re.search(pattern, text(info.get("method_description"))))

    return False


def method_summary(info: JsonDict) -> str:
    lines: list[str] = []
    for line in text(info.get("method_description")).replace('"""', "").replace("'''", "").splitlines():
        line = line.strip()
        if line and not line.startswith(("def ", ":param", ":return:", ">>>")):
            lines.append(line)

    return " ".join(lines).strip() or "Implements the visible task behavior."


def dependencies(info: JsonDict) -> JsonDict:
    deps = as_dict(info.get("dependencies"))
    reads = as_list(deps.get("field_dependencies"))

    return {
        "reads": reads,
        "modifies": [],
        "preserves": reads,
        "calls": as_list(deps.get("method_dependencies")),
        "uses_libraries": as_list(deps.get("lib_dependencies")),
    }


def default_invalid_behavior() -> JsonDict:
    return {
        "specified": False,
        "expected_behavior": "",
        "exception_type": "",
        "description": "",
        "source": "",
    }


def default_body(summary: str) -> JsonDict:
    return {
        "interface": {"inputs": [], "output": {"type": "", "description": summary}},
        "preconditions": [],
        "postconditions": [summary],
        "invariants": [],
        "edge_cases": [],
        "invalid_input_behavior": default_invalid_behavior(),
    }


def metadata_contract(task: JsonDict) -> JsonDict:
    methods: list[JsonDict] = []

    for info in as_list(task.get("methods_info")):
        if not isinstance(info, dict):
            continue

        name = text(info.get("method_name"))
        if not name:
            continue

        methods.append(
            {
                "method_name": name,
                "signature": method_signature(info),
                "is_static": method_is_static(task, name),
                "dependencies": dependencies(info),
                "contract": default_body(method_summary(info)),
            }
        )

    return {
        "task": {
            "task_id": task_id(task),
            "benchmark": "ClassEval",
            "language": "python",
            "execution_model": "class_level",
            "class_name": safe_class_name(task),
            "entry_point": safe_class_name(task),
            "summary": text(task.get("class_description")),
        },
        "class_interface": {
            "fields": as_list(task.get("fields")),
            "methods": [m["method_name"] for m in methods],
        },
        "constructor": {
            "signature": constructor_signature(task),
            "initializes": as_list(task.get("fields")),
            "contract": default_body("Initializes the class state required by the class."),
        },
        "class_invariants": [],
        "method_contracts": methods,
        "interaction_contracts": [],
    }


def normalize_body(body: JsonDict, fallback_body: JsonDict) -> JsonDict:
    body = body.copy()

    interface = as_dict(body.get("interface")).copy()
    output = as_dict(interface.get("output")).copy()

    fallback_interface = as_dict(fallback_body.get("interface"))
    fallback_output = as_dict(fallback_interface.get("output"))

    output["type"] = text(output.get("type")) or text(fallback_output.get("type"))
    output["description"] = text(output.get("description")) or text(fallback_output.get("description"))

    interface["inputs"] = as_list(interface.get("inputs"))
    interface["output"] = output

    body["interface"] = interface
    body["preconditions"] = as_list(body.get("preconditions"))
    body["postconditions"] = as_list(body.get("postconditions")) or as_list(fallback_body.get("postconditions"))
    body["invariants"] = as_list(body.get("invariants"))
    body["edge_cases"] = as_list(body.get("edge_cases"))
    body["invalid_input_behavior"] = as_dict(body.get("invalid_input_behavior")) or default_invalid_behavior()

    return body


def repair_contract(contract: JsonDict, task: JsonDict) -> JsonDict:
    contract = normalize_contract_shape(contract)
    fallback = metadata_contract(task)

    task_block = as_dict(contract.get("task"))
    task_block.update(fallback["task"])
    contract["task"] = task_block
    contract["class_interface"] = fallback["class_interface"]

    constructor = as_dict(contract.get("constructor")).copy()
    constructor["signature"] = fallback["constructor"]["signature"]
    constructor["initializes"] = as_list(constructor.get("initializes")) or fallback["constructor"]["initializes"]
    constructor["contract"] = normalize_body(
        as_dict(constructor.get("contract")) or fallback["constructor"]["contract"],
        fallback["constructor"]["contract"],
    )
    contract["constructor"] = constructor

    existing = {
        text(item.get("method_name")): item
        for item in as_list(contract.get("method_contracts"))
        if isinstance(item, dict) and text(item.get("method_name"))
    }

    repaired: list[JsonDict] = []
    for fallback_method in fallback["method_contracts"]:
        name = fallback_method["method_name"]
        method = as_dict(existing.get(name)).copy()
        method["method_name"] = name
        method["signature"] = fallback_method["signature"]
        method["is_static"] = fallback_method["is_static"]
        method["dependencies"] = as_dict(method.get("dependencies")) or fallback_method["dependencies"]
        method["contract"] = normalize_body(as_dict(method.get("contract")), fallback_method["contract"])
        repaired.append(method)

    contract["method_contracts"] = repaired
    contract["class_invariants"] = as_list(contract.get("class_invariants"))
    contract["interaction_contracts"] = as_list(contract.get("interaction_contracts"))

    return contract


def cleanup_json_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"```$", "", value).strip()
    value = value.replace("\ufeff", "")
    value = re.sub(r",\s*([}\]])", r"\1", value)
    return value


def json_candidates(value: str) -> list[str]:
    value = cleanup_json_text(value)
    candidates: list[str] = []

    start = -1
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(value):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(value[start : index + 1])

    if value.startswith("{") and value not in candidates:
        candidates.insert(0, value)

    return candidates


def robust_json_loads(value: str) -> JsonDict:
    errors: list[str] = []

    for candidate in json_candidates(value):
        try:
            obj = json.loads(cleanup_json_text(candidate))
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError("No valid JSON object found in model response. " + " | ".join(errors[:3]))


def parse_contract_response(response: str, task: JsonDict) -> tuple[JsonDict, list[str]]:
    contract = repair_contract(robust_json_loads(response), task)
    errors = stage2_shape_errors(contract, expected_methods=method_names(task))
    return contract, errors


def information_score(contract: JsonDict) -> int:
    score = len(as_list(contract.get("class_invariants"))) + len(as_list(contract.get("interaction_contracts")))

    for method in as_list(contract.get("method_contracts")):
        method = as_dict(method)
        body = as_dict(method.get("contract"))
        interface = as_dict(body.get("interface"))
        output = as_dict(interface.get("output"))
        deps = as_dict(method.get("dependencies"))

        score += int(bool(text(method.get("signature"))))
        score += int(bool(as_list(interface.get("inputs"))))
        score += int(bool(text(output.get("type"))))
        score += 2 * int(bool(text(output.get("description"))))
        score += 3 * int(bool(as_list(body.get("postconditions"))))
        score += 2 * int(bool(as_list(body.get("edge_cases"))))
        score += sum(int(bool(as_list(deps.get(k)))) for k in ("reads", "modifies", "preserves", "calls", "uses_libraries"))

    return score


def feedback_from_records(evaluation: JsonDict, generation: JsonDict | None = None) -> JsonDict:
    prepared = evaluation.get("feedback_for_next_contract")

    if isinstance(prepared, dict) and prepared:
        source = prepared
    else:
        metrics = as_dict(as_dict(evaluation.get("evaluation")).get("metrics"))
        source = {
            "passed": evaluation.get("passed") is True,
            "failure_type": text(evaluation.get("failure_type")),
            "tests_passed": metrics.get("passed"),
            "tests_total": metrics.get("total"),
            "tests_failed": metrics.get("failed"),
            "failures": metrics.get("failures"),
            "errors": metrics.get("errors"),
            "generation_error": as_dict(generation or {}).get("error"),
        }

    allowed = ("passed", "failure_type", "tests_passed", "tests_total", "tests_failed", "failures", "errors", "skipped")
    return {k: source.get(k) for k in allowed if source.get(k) not in (None, "", [], {})}


def load_previous_contract(
    task: JsonDict,
    provider: str,
    model: str,
    base_provider: str,
    base_model: str,
    use_previous_optimized: bool = False,
) -> tuple[JsonDict, str, str | None]:
    if use_previous_optimized:
        path = optimized_path(task, provider, model)
        if path.exists():
            record = load_json(path)
            if record.get("status") == "success" and record.get("fallback_to_previous_contract") is not True and isinstance(record.get("contract"), dict):
                return repair_contract(record["contract"], task), "previous_optimized_contract", str(path)

    path = v1_contract_path(task, base_provider, base_model)
    if path.exists():
        record = load_json(path)
        if record.get("status") == "success" and isinstance(record.get("contract"), dict):
            return repair_contract(record["contract"], task), "original_stage2_contract", str(path)

    return metadata_contract(task), "metadata_task_contract", None


def load_previous_feedback(
    task: JsonDict,
    provider: str,
    model: str,
    base_provider: str,
    base_model: str,
    use_previous_optimized: bool = False,
) -> tuple[JsonDict, str, str | None]:
    if use_previous_optimized:
        opt_eval = optimized_evaluation_path(task, provider, model)
        opt_gen = optimized_generation_path(task, provider, model)
        if opt_eval.exists():
            return (
                feedback_from_records(load_json(opt_eval), load_json(opt_gen) if opt_gen.exists() else None),
                "previous_optimized_evaluation",
                str(opt_eval),
            )

    v1_eval = v1_evaluation_path(task, base_provider, base_model)
    v1_gen = v1_generation_path(task, base_provider, base_model)

    if v1_eval.exists():
        return (
            feedback_from_records(load_json(v1_eval), load_json(v1_gen) if v1_gen.exists() else None),
            "original_contract_guided_evaluation",
            str(v1_eval),
        )

    return {}, "no_previous_feedback", None


def optimization_mode(contract_source: str, feedback: JsonDict) -> str:
    if contract_source == "metadata_task_contract":
        return "synthesis"
    return "strengthening" if feedback.get("passed") is True else "correction"


def mode_instruction(mode: str) -> str:
    if mode == "synthesis":
        return "No usable v1 contract exists. Generate a complete concise contract from the visible task."
    if mode == "strengthening":
        return "The v1 contract-guided code passed. Preserve behavior. Only clarify vague or missing clauses."
    return "The v1 contract-guided code failed. Correct the contract using summarized feedback, without test-specific patches."


def prompt_contract_template(task: JsonDict) -> JsonDict:
    template = metadata_contract(task)

    for method in as_list(template.get("method_contracts")):
        body = method["contract"]
        body["preconditions"] = []
        body["postconditions"] = ["..."]
        body["invariants"] = []
        body["edge_cases"] = []

    return template


def pass_ratio(feedback: JsonDict) -> float:
    passed = feedback.get("tests_passed")
    total = feedback.get("tests_total")

    if isinstance(passed, int) and isinstance(total, int) and total > 0:
        return max(0.0, min(1.0, passed / total))

    return 1.0 if feedback.get("passed") is True else 0.0


def clause_texts(contract: JsonDict) -> list[str]:
    result: list[str] = []

    for method in as_list(contract.get("method_contracts")):
        body = as_dict(as_dict(method).get("contract"))
        for key in ("preconditions", "postconditions", "invariants", "edge_cases"):
            result.extend(text(x) for x in as_list(body.get(key)) if text(x))

    result.extend(text(x) for x in as_list(contract.get("class_invariants")) if text(x))
    return result


def state_features(task: JsonDict, contract: JsonDict, feedback: JsonDict) -> JsonDict:
    methods = as_list(contract.get("method_contracts"))
    method_count = max(1, len(method_names(task)))
    clauses = clause_texts(contract)

    vague_terms = ("correct", "proper", "valid", "invalid", "handle", "appropriate", "as expected", "etc")
    vague = sum(1 for clause in clauses if len(clause.split()) < 3 or any(term in clause.lower() for term in vague_terms))

    failure = text(feedback.get("failure_type")).lower()
    generation_fail = any(term in failure for term in ("missing", "signature", "syntax", "import", "generation"))
    runtime_fail = "runtime" in failure or "error" in failure
    logical_fail = "logical" in failure or "assert" in failure or "fail" in failure
    syntax_fail = "syntax" in failure
    timeout_fail = "timeout" in failure

    postconditions = 0
    edge_cases = 0

    for method in methods:
        body = as_dict(as_dict(method).get("contract"))
        postconditions += len(as_list(body.get("postconditions")))
        edge_cases += len(as_list(body.get("edge_cases")))

    return {
        "bias": 1.0,
        "method_count": min(1.0, method_count / 12),
        "field_count": min(1.0, len(as_list(task.get("fields"))) / 12),
        "contract_coverage": min(1.0, len(methods) / method_count),
        "avg_postconditions": min(1.0, (postconditions / max(1, len(methods))) / 6),
        "avg_edge_cases": min(1.0, (edge_cases / max(1, len(methods))) / 5),
        "class_invariant_count": min(1.0, len(as_list(contract.get("class_invariants"))) / 8),
        "vague_ratio": 0.0 if not clauses else vague / len(clauses),
        "prev_passed": 1.0 if feedback.get("passed") is True else 0.0,
        "pass_ratio": pass_ratio(feedback),
        "failed_generation": 1.0 if generation_fail else 0.0,
        "failed_logical": 1.0 if logical_fail else 0.0,
        "failed_runtime": 1.0 if runtime_fail else 0.0,
        "failed_syntax": 1.0 if syntax_fail else 0.0,
        "failed_timeout": 1.0 if timeout_fail else 0.0,
        "failed_other": 1.0 if failure and not any([generation_fail, logical_fail, runtime_fail, syntax_fail, timeout_fail]) else 0.0,
    }


def vector(features: JsonDict) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def fresh_policy() -> JsonDict:
    width = len(FEATURE_NAMES)
    return {
        "stage": "2R_contextual_bandit_policy",
        "policy_type": "contextual_bandit_softmax",
        "policy_version": "contextual_bandit_in_3_files_v1",
        "created_at": now(),
        "updated_at": now(),
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
    }


def load_policy(args: argparse.Namespace, model: str) -> tuple[JsonDict, str]:
    path = Path(args.policy_file) if args.policy_file else default_policy_path(args.provider, model)
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
    policy["epsilon"] = float(policy.get("epsilon", args.bandit_epsilon))
    policy["temperature"] = float(policy.get("temperature", args.bandit_temperature))

    return policy, str(path)


def load_policy_text(args: argparse.Namespace, model: str) -> tuple[str, str | None]:
    if not args.use_policy and not args.policy_file:
        return "", None

    path = Path(args.policy_file) if args.policy_file else default_policy_path(args.provider, model)
    if not path.exists():
        return "", str(path)

    data = load_json(path)
    return text(data.get("optimized_policy_text")), str(path)


def action_by_id(action_id: str) -> JsonDict:
    for action in ACTIONS:
        if action["id"] == action_id:
            return dict(action)
    return dict(ACTIONS[-1])


def softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    temperature = max(1e-6, float(temperature))
    top = max(scores.values()) if scores else 0.0
    exps = {key: math.exp((value - top) / temperature) for key, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def select_action(policy: JsonDict, features: JsonDict, args: argparse.Namespace) -> JsonDict:
    rng = random.Random(args.bandit_seed + abs(hash(dump(features))) % 100000)
    x = vector(features)
    weights = as_dict(policy.get("weights"))

    scores = {
        action_id: sum(weight * value for weight, value in zip(as_list(weights.get(action_id)), x))
        for action_id in ACTION_IDS
    }

    probs = softmax(scores, float(policy.get("temperature", args.bandit_temperature)))
    explored = False

    if args.greedy_action:
        action_id = max(probs, key=probs.get)
        mode = "greedy"
    elif rng.random() < float(policy.get("epsilon", args.bandit_epsilon)):
        action_id = rng.choice(ACTION_IDS)
        explored = True
        mode = "epsilon"
    else:
        draw = rng.random()
        cumulative = 0.0
        action_id = ACTION_IDS[-1]

        for candidate in ACTION_IDS:
            cumulative += probs[candidate]
            if draw <= cumulative:
                action_id = candidate
                break

        mode = "softmax_sample"

    return {
        "policy_version": text(policy.get("policy_version")) or "contextual_bandit_in_3_files_v1",
        "action_id": action_id,
        "selected_action": action_by_id(action_id),
        "action_scores": scores,
        "action_probabilities": probs,
        "selection_mode": mode,
        "explored": explored,
        "feature_names": FEATURE_NAMES,
    }


def make_prompt(
    task: JsonDict,
    previous: JsonDict,
    feedback: JsonDict,
    contract_source: str,
    feedback_source: str,
    mode: str,
    policy_text: str,
    action_selection: JsonDict,
) -> str:
    selected = as_dict(action_selection.get("selected_action"))

    parts = [
        "Return exactly ONE valid minified JSON object. No markdown. No prose. No code.",
        "The JSON object is an optimized Stage-2 Design-by-Contract specification for one ClassEval Python class.",
        "Keep the contract concise so the JSON is not truncated.",
        "Use contracts as guidance for later code generation; do not override the visible skeleton.",
        "",
        "OUTPUT_TEMPLATE_SHAPE:",
        dump(prompt_contract_template(task)),
        "",
        "SELECTED_RL_ACTION:",
        dump({"id": selected.get("id"), "name": selected.get("name"), "instruction": selected.get("instruction")}),
        "",
        "ACTION_REQUIREMENT:",
        "Use the selected RL action as the primary revision strategy. It must directly affect which clauses you add, remove, or clarify.",
        "",
        "OPTIMIZATION_MODE:",
        mode,
        "",
        "MODE_INSTRUCTION:",
        mode_instruction(mode),
        "",
        "VISIBLE_TASK:",
        dump(task_view(task)),
        "",
        "PREVIOUS_CONTRACT_SOURCE:",
        contract_source,
        "",
        "PREVIOUS_CONTRACT:",
        dump(previous),
        "",
        "PREVIOUS_FEEDBACK_SOURCE:",
        feedback_source,
        "",
        "PREVIOUS_FEEDBACK_SUMMARY:",
        dump(feedback),
    ]

    if policy_text:
        parts.extend(["", "LEARNED_POLICY_NOTE:", policy_text[:2000]])

    parts.extend(
        [
            "",
            "RULES:",
            "- Output JSON only.",
            "- Preserve visible class name, constructor, fields, imports, method names, signatures, and staticmethod status exactly.",
            "- Cover every visible method exactly once.",
            "- Preserve useful v1 behavior unless feedback clearly shows it is wrong.",
            "- Do not invent exceptions or strict invalid-input behavior unless supported by the visible task or feedback.",
            "- Do not include hidden tests, reference solutions, expected answers, assertions, exact tracebacks, or test names.",
            "- Do not write test-specific patches.",
            "- Prefer concise executable postconditions and edge cases over vague prose.",
        ]
    )

    prompt = "\n".join(parts)
    ensure_no_reference_leak(task, prompt)
    return prompt


def call_contract(client: Any, model: str, prompt: str, task: JsonDict, args: argparse.Namespace) -> tuple[JsonDict, str, JsonDict | None, list[str]]:
    response, api_result = call_chat_model(
        client=client,
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        json_mode=True,
    )

    contract, errors = parse_contract_response(response, task)

    if errors and args.reject_shape_errors:
        raise ValueError("Stage 2 shape errors: " + "; ".join(errors[:10]))

    return contract, response, api_result, errors


def make_record(task: JsonDict, args: argparse.Namespace, model: str, **extra: Any) -> JsonDict:
    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
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


def generate_one(task: JsonDict, args: argparse.Namespace, model: str, client: Any) -> JsonDict:
    base_provider = args.base_provider or args.provider
    base_model = args.base_model or model

    previous, contract_source, contract_file = load_previous_contract(
        task,
        args.provider,
        model,
        base_provider,
        base_model,
        args.use_previous_optimized,
    )

    feedback, feedback_source, feedback_file = load_previous_feedback(
        task,
        args.provider,
        model,
        base_provider,
        base_model,
        args.use_previous_optimized,
    )

    features = state_features(task, previous, feedback)
    policy, policy_file = load_policy(args, model)
    selection = select_action(policy, features, args)
    policy_text, text_policy_file = load_policy_text(args, model)

    mode = optimization_mode(contract_source, feedback)
    prompt = make_prompt(task, previous, feedback, contract_source, feedback_source, mode, policy_text, selection)

    try:
        contract, response, api_result, errors = call_contract(client, model, prompt, task, args)
        status = "success"
        failure_type = None
        failure_detail = None
        error = None
        fallback = False
        version = "v2_rl_optimized"
    except Exception as exc:
        logger.exception("Optimized contract model call failed; saving fallback record: %s", task_id(task))
        failure_type, failure_detail = classify_contract_error(exc)
        contract = previous
        response = ""
        api_result = None
        errors = [failure_detail]
        status = "fallback"
        error = failure_detail
        fallback = True
        version = "fallback_previous_contract"

    return make_record(
        task,
        args,
        model,
        status=status,
        failure_type=failure_type,
        failure_detail=failure_detail,
        contract=contract,
        contract_type="rl_optimized",
        contract_version=version,
        optimization_mode=mode,
        contract_source=contract_source,
        feedback_source=feedback_source,
        previous_contract_file=contract_file,
        previous_feedback_file=feedback_file,
        previous_information_score=information_score(previous),
        information_score=information_score(contract),
        shape_errors=errors,
        feedback=feedback,
        fallback_to_previous_contract=fallback,
        excluded_from_optimized_metrics=fallback or status != "success",
        policy_used=True,
        policy_file=text_policy_file or policy_file,
        rl_policy={
            "policy_type": "contextual_bandit_softmax",
            "policy_file": policy_file,
            "policy_version": selection.get("policy_version"),
            "state_features": features,
            "selected_action": selection.get("selected_action"),
            "action_id": selection.get("action_id"),
            "action_probabilities": selection.get("action_probabilities"),
            "action_scores": selection.get("action_scores"),
            "selection_mode": selection.get("selection_mode"),
            "explored": selection.get("explored"),
            "feature_names": FEATURE_NAMES,
        },
        error=error,
        **({"prompt": prompt, "raw_response": response, "api_result": api_result} if args.debug else {}),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feedback-optimized ClassEval contracts with embedded contextual-bandit RL.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-provider")
    parser.add_argument("--base-model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-policy", action="store_true")
    parser.add_argument("--policy-file")
    parser.add_argument("--use-previous-optimized", action="store_true")
    parser.add_argument("--reject-shape-errors", action="store_true")
    parser.add_argument("--greedy-action", action="store_true")
    parser.add_argument("--bandit-seed", type=int, default=0)
    parser.add_argument("--bandit-epsilon", type=float, default=0.15)
    parser.add_argument("--bandit-temperature", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    client = get_client(args.provider)
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = 0
    failed = 0
    skipped = 0
    fallback = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        path = optimized_path(task, args.provider, model)

        if path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}] SKIP {task_id(task)}")
            continue

        try:
            record = generate_one(task, args, model, client)
        except Exception as exc:
            logger.exception("Optimized contract failed: %s", task_id(task))
            failure_type, failure_detail = classify_contract_error(exc)
            record = make_record(
                task,
                args,
                model,
                status="failed",
                failure_type=failure_type,
                failure_detail=failure_detail,
                contract=None,
                contract_type="rl_optimized",
                contract_version="failed_optimization",
                fallback_to_previous_contract=False,
                excluded_from_optimized_metrics=True,
                error=failure_detail,
            )

        save_json(path, record)

        if record["status"] == "success":
            done += 1
            print(f"[{index}] DONE {task_id(task)} action={(record.get('rl_policy') or {}).get('action_id')}")
        elif record["status"] == "fallback":
            fallback += 1
            print(f"[{index}] FALLBACK/{record.get('failure_type')} {task_id(task)}")
        else:
            failed += 1
            print(f"[{index}] FAILED/{record.get('failure_type')} {task_id(task)}")

        if args.delay:
            time.sleep(args.delay)

    print("\nClassEval feedback-optimized contract generation finished.")
    print(f"Completed: {done}")
    print(f"Fallback-to-previous: {fallback}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()