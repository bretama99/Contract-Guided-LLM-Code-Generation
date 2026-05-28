from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.classeval.core import JsonDict, as_dict, task_id, text
from src.classeval.generate_contracts import TASK_FILE
from src.classeval.rl_generate_optimized_contracts import safe_class_name
from src.classeval.rl_generate_optimized_contracts import optimized_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, load_json_list, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model
from src.common.task_utils import select_tasks

STAGE = "2R_revision_preference_building"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_revision_preferences"
LOG_FILE = LOG_ROOT / "classeval_rl_revision_preferences.log"


def preferences_path(provider: str, model: str) -> Path:
    return OUT_DIR / safe_name(provider) / safe_name(model) / "preferences.json"


def find_json(root: Path, provider: str, model: str, task: JsonDict, must_contain: tuple[str, ...]) -> Path | None:
    provider_key = safe_name(provider)
    model_key = safe_name(model)
    task_key = safe_name(task_id(task))

    if not root.exists():
        return None

    for path in root.rglob("*.json"):
        lower = str(path).lower()
        if provider_key not in path.parts or model_key not in path.parts:
            continue
        if task_key not in str(path):
            continue
        if all(term in lower for term in must_contain):
            return path

    return None


def v1_contract_path(task: JsonDict, provider: str, model: str) -> Path | None:
    return find_json(
        OUTPUT_ROOT / "classeval",
        provider,
        model,
        task,
        must_contain=("contract",),
    )


def v1_eval_path(task: JsonDict, provider: str, model: str) -> Path | None:
    return find_json(
        OUTPUT_ROOT / "classeval" / "evaluation",
        provider,
        model,
        task,
        must_contain=("contract",),
    )


def v2_eval_path(task: JsonDict, provider: str, model: str) -> Path | None:
    return find_json(
        OUTPUT_ROOT / "classeval" / "evaluation" / "rl_optimized_contract_guided",
        provider,
        model,
        task,
        must_contain=(),
    )


def load_contract(path: Path | None) -> JsonDict | None:
    if path is None or not path.exists():
        return None
    data = load_json(path)
    contract = data.get("contract")
    return contract if isinstance(contract, dict) else None


def reward_from_eval(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0

    data = load_json(path)
    if data.get("passed") is True:
        return 1.0

    metrics = as_dict(as_dict(data.get("evaluation")).get("metrics"))
    passed = metrics.get("passed")
    total = metrics.get("total")

    if isinstance(passed, int) and isinstance(total, int) and total > 0:
        return max(0.0, min(1.0, passed / total))

    return 0.0


def compact_prompt(task: JsonDict, feedback: JsonDict | None) -> str:
    payload = {
        "class_name": safe_class_name(task),
        "task_id": task_id(task),
        "feedback": feedback or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def make_preference(task: JsonDict, provider: str, model: str, min_gap: float) -> JsonDict | None:
    v1_contract_file = v1_contract_path(task, provider, model)
    v2_contract_file = optimized_path(task, provider, model)
    v1_eval_file = v1_eval_path(task, provider, model)
    v2_eval_file = v2_eval_path(task, provider, model)

    v1_contract = load_contract(v1_contract_file)
    v2_contract = load_contract(v2_contract_file)

    if not v1_contract or not v2_contract:
        return None

    r1 = reward_from_eval(v1_eval_file)
    r2 = reward_from_eval(v2_eval_file)
    gap = abs(r2 - r1)

    if gap < min_gap:
        return None

    chosen_contract, rejected_contract = (v2_contract, v1_contract) if r2 > r1 else (v1_contract, v2_contract)
    chosen_round, rejected_round = ("v2", "v1") if r2 > r1 else ("v1", "v2")

    v2_record = load_json(v2_contract_file) if v2_contract_file.exists() else {}
    feedback = as_dict(v2_record.get("feedback"))

    return {
        "task_id": task_id(task),
        "class_name": safe_class_name(task),
        "stage": STAGE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": compact_prompt(task, feedback),
        "chosen": json.dumps(chosen_contract, ensure_ascii=False, indent=2),
        "rejected": json.dumps(rejected_contract, ensure_ascii=False, indent=2),
        "chosen_contract": chosen_contract,
        "rejected_contract": rejected_contract,
        "chosen_round": chosen_round,
        "rejected_round": rejected_round,
        "v1_reward": round(r1, 6),
        "v2_reward": round(r2, 6),
        "reward_gap": round(gap, 6),
        "v1_contract_file": None if v1_contract_file is None else str(v1_contract_file),
        "v2_contract_file": str(v2_contract_file),
        "v1_eval_file": None if v1_eval_file is None else str(v1_eval_file),
        "v2_eval_file": None if v2_eval_file is None else str(v2_eval_file),
        "reason": "Chosen contract achieved higher reward in sequential contract refinement.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preferences from v1/v2 iterative contract refinement.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--min-gap", type=float, default=0.01)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    out = preferences_path(args.provider, model)

    if out.exists() and not args.overwrite:
        print(f"Preference file already exists: {out}")
        return

    tasks = select_tasks(load_json_list(TASK_FILE), start=args.start, count=args.count)
    preferences: list[JsonDict] = []
    skipped = 0
    started = time.perf_counter()

    for task in tasks:
        pref = make_preference(task, args.provider, model, args.min_gap)
        if pref:
            preferences.append(pref)
        else:
            skipped += 1

    save_json(
        out,
        {
            "stage": STAGE,
            "provider": args.provider,
            "model": model,
            "min_gap": args.min_gap,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": len(preferences),
            "skipped": skipped,
            "preferences": preferences,
        },
    )

    print("\nClassEval revision preference building finished.")
    print(f"Preferences: {len(preferences)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {out}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()