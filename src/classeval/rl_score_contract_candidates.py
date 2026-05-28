from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.core import JsonDict, as_dict, as_list, task_id, text
from src.classeval.execution import evaluate_generation, failure_record
from src.classeval.generate_contracts import TASK_FILE
from src.classeval.rl_generate_contract_candidates import candidate_path
from src.classeval.rl_generate_from_contract_candidates import generation_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model
from src.common.task_utils import select_tasks

STAGE = "2R_contract_candidate_reward"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_contract_candidate_scores"
LOG_FILE = LOG_ROOT / "classeval_rl_contract_candidate_scores.log"

WEIGHTS = {
    "downstream": 0.75,
    "method_coverage": 0.07,
    "guarantee_coverage": 0.08,
    "dependency_coverage": 0.05,
    "interface_coverage": 0.05,
}

OVERCONSTRAINTS = (
    r"\bnon-empty\b",
    r"\bnot empty\b",
    r"\bmust be positive\b",
    r"\bstrictly positive\b",
    r"\bmust be sorted\b",
    r"\balready sorted\b",
    r"\bmust be unique\b",
    r"\bnon-null\b",
    r"\bnot none\b",
    r"\bfinite\b",
)

WEAK_PRECONDITIONS = (
    r"\bexpected to be a\b",
    r"\bexpected to be an\b",
    r"\bmust be a dict\b",
    r"\bmust be a str\b",
    r"\bmust be an int\b",
    r"\bmust be a float\b",
    r"\bmust be a bool\b",
)


def safe_class_name(task: JsonDict) -> str:
    name = text(task.get("class_name"))
    if name:
        return name

    for line in text(task.get("skeleton")).splitlines():
        line = line.strip()
        if line.startswith("class ") and ":" in line:
            return line.removeprefix("class ").split(":", 1)[0].split("(", 1)[0].strip()

    return text(task.get("entry_point")) or "UnknownClass"


def safe_method_names(task: JsonDict) -> list[str]:
    names = [text(name) for name in as_list(task.get("method_names")) if text(name)]
    if names:
        return names

    return [
        text(item.get("method_name"))
        for item in as_list(task.get("methods_info"))
        if isinstance(item, dict) and text(item.get("method_name"))
    ]


def score_path(task: JsonDict, cp: str, cm: str, gp: str, gm: str, cid: int) -> Path:
    return (
        OUT_DIR
        / safe_name(cp)
        / safe_name(cm)
        / safe_name(gp)
        / safe_name(gm)
        / safe_name(task_id(task))
        / f"candidate_{cid}.json"
    )


def load_required(path: Path, label: str) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")

    return data


def load_candidate(task: JsonDict, provider: str, model: str, cid: int) -> JsonDict:
    return load_required(candidate_path(task, provider, model, cid), "candidate")


def load_generation(task: JsonDict, provider: str, model: str, cid: int) -> tuple[JsonDict, Path]:
    path = generation_path(task, provider, model, cid)
    return load_required(path, "candidate-guided generation"), path


def all_text(value: Any) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            parts.append(item)

    walk(value)
    return "\n".join(parts).lower()


def eval_counts(evaluation: JsonDict) -> tuple[int | None, int | None]:
    metrics = as_dict(as_dict(evaluation.get("evaluation")).get("metrics"))
    passed = metrics.get("passed")
    total = metrics.get("total")

    if isinstance(passed, int) and isinstance(total, int):
        return passed, total

    return None, None


def downstream_score(evaluation: JsonDict) -> float:
    if evaluation.get("passed") is True:
        return 1.0

    passed, total = eval_counts(evaluation)
    if total and passed is not None:
        return 0.5 * max(0.0, min(1.0, passed / total))

    return 0.0


def contract_methods(contract: JsonDict) -> list[JsonDict]:
    return [method for method in as_list(contract.get("method_contracts")) if isinstance(method, dict)]


def method_coverage(contract: JsonDict, task: JsonDict) -> float:
    expected = set(safe_method_names(task))
    actual = {text(method.get("method_name")) for method in contract_methods(contract)}

    if not expected:
        return 1.0

    return len(expected & actual) / len(expected)


def ratio(contract: JsonDict, predicate) -> float:
    methods = contract_methods(contract)
    if not methods:
        return 0.0

    return sum(1 for method in methods if predicate(method)) / len(methods)


def guarantee_coverage(contract: JsonDict) -> float:
    return ratio(
        contract,
        lambda method: bool(as_list(as_dict(method.get("contract")).get("postconditions"))),
    )


def dependency_coverage(contract: JsonDict) -> float:
    def has_dependency(method: JsonDict) -> bool:
        deps = as_dict(method.get("dependencies"))
        return any(
            as_list(deps.get(key))
            for key in ("reads", "modifies", "preserves", "calls", "uses_libraries")
        )

    return ratio(contract, has_dependency)


def interface_coverage(contract: JsonDict) -> float:
    def has_output(method: JsonDict) -> bool:
        contract_body = as_dict(method.get("contract"))
        interface = as_dict(contract_body.get("interface"))
        output = as_dict(interface.get("output"))
        return bool(text(output.get("description")) or text(output.get("type")))

    return ratio(contract, has_output)


def regex_count(patterns: tuple[str, ...], value: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, value, flags=re.I))


def weak_precondition_penalty(contract: JsonDict) -> float:
    hits = 0

    for method in contract_methods(contract):
        body = as_dict(method.get("contract"))
        for clause in as_list(body.get("preconditions")):
            hits += regex_count(WEAK_PRECONDITIONS, text(clause))

    return min(0.20, 0.04 * hits)


def overconstraint_penalty(contract: JsonDict, task: JsonDict) -> float:
    contract_text = all_text(contract)
    task_text = all_text(task)

    hits = sum(
        1
        for pattern in OVERCONSTRAINTS
        if re.search(pattern, contract_text, re.I) and not re.search(pattern, task_text, re.I)
    )

    return min(0.25, 0.05 * hits)


def missing_guarantee_penalty(contract: JsonDict) -> float:
    missing = sum(
        1
        for method in contract_methods(contract)
        if not as_list(as_dict(method.get("contract")).get("postconditions"))
    )

    return min(0.25, 0.05 * missing)


def failure_penalty(evaluation: JsonDict) -> float:
    if evaluation.get("passed") is True:
        return 0.0

    failure_type = text(evaluation.get("failure_type"))

    if failure_type in {"missing_generation", "missing_or_invalid_probe", "generation_failed"}:
        return 0.40

    return 0.25


def compute_reward(candidate: JsonDict, task: JsonDict, evaluation: JsonDict) -> JsonDict:
    contract = as_dict(candidate.get("contract"))

    components = {
        "downstream": downstream_score(evaluation),
        "method_coverage": method_coverage(contract, task),
        "guarantee_coverage": guarantee_coverage(contract),
        "dependency_coverage": dependency_coverage(contract),
        "interface_coverage": interface_coverage(contract),
    }

    penalties = {
        "failure": failure_penalty(evaluation),
        "overconstraint": overconstraint_penalty(contract, task),
        "weak_preconditions": weak_precondition_penalty(contract),
        "missing_guarantees": missing_guarantee_penalty(contract),
    }

    total = sum(WEIGHTS[name] * components[name] for name in WEIGHTS)
    total -= sum(penalties.values())
    total = max(0.0, min(1.0, total))

    if evaluation.get("passed") is True:
        total = max(0.50, total)
    else:
        total = min(0.49, total)

    return {
        "total": round(total, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "penalties": {key: round(value, 6) for key, value in penalties.items()},
        "weights": WEIGHTS,
    }


def evaluate_probe(
    task: JsonDict,
    generation: JsonDict,
    path: Path,
    provider: str,
    model: str,
    timeout: float,
    include_code: bool,
) -> JsonDict:
    return evaluate_generation(
        task,
        generation,
        path,
        provider=provider,
        model=model,
        stage=STAGE,
        timeout=timeout,
        include_code=include_code,
    )


def failed_eval(task: JsonDict, provider: str, model: str, error: str) -> JsonDict:
    return failure_record(
        task,
        provider=provider,
        model=model,
        stage=STAGE,
        failure_type="missing_or_invalid_probe",
        error=error,
    )


def make_record(
    task: JsonDict,
    args: argparse.Namespace,
    candidate: JsonDict,
    evaluation: JsonDict,
    reward: JsonDict,
    cid: int,
    candidate_model: str,
    generation_provider: str,
    generation_model: str,
) -> JsonDict:
    passed, total = eval_counts(evaluation)

    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
        "stage": STAGE,
        "status": "success",
        "candidate_id": cid,
        "candidate_policy": candidate.get("candidate_policy"),
        "candidate_provider": args.provider,
        "candidate_model": candidate_model,
        "generation_provider": generation_provider,
        "generation_model": generation_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "passed": evaluation.get("passed") is True,
            "failure_type": evaluation.get("failure_type"),
            "tests": {
                "passed": passed,
                "total": total,
            },
        },
        "reward": reward,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ClassEval RL Stage 2 contract candidates.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--generation-provider", choices=sorted(PROVIDERS))
    parser.add_argument("--generation-model")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    candidate_model = args.model or default_model(args.provider)
    generation_provider = args.generation_provider or args.provider
    generation_model = args.generation_model or candidate_model
    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)

    done = skipped = 0
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=args.start):
        for cid in range(args.num_candidates):
            out_path = score_path(
                task,
                args.provider,
                candidate_model,
                generation_provider,
                generation_model,
                cid,
            )

            if out_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[{index}] SKIP {task_id(task)} candidate_{cid}")
                continue

            try:
                candidate = load_candidate(task, args.provider, candidate_model, cid)
                generation, generation_file = load_generation(task, generation_provider, generation_model, cid)
                evaluation = evaluate_probe(
                    task,
                    generation,
                    generation_file,
                    generation_provider,
                    generation_model,
                    args.timeout,
                    args.include_code,
                )
            except Exception as exc:
                candidate = {"candidate_policy": None, "contract": {}, "error": str(exc)}
                evaluation = failed_eval(task, generation_provider, generation_model, str(exc))

            reward = compute_reward(candidate, task, evaluation)
            save_json(
                out_path,
                make_record(
                    task,
                    args,
                    candidate,
                    evaluation,
                    reward,
                    cid,
                    candidate_model,
                    generation_provider,
                    generation_model,
                ),
            )

            done += 1
            status = "PASS" if evaluation.get("passed") else "FAIL"
            print(f"[{index}] {status} {task_id(task)} candidate_{cid} reward={reward['total']}")

    print("\nClassEval RL Stage 2 contract candidate scoring finished.")
    print(f"Scored: {done}")
    print(f"Skipped: {skipped}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()