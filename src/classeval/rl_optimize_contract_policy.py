from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.classeval.rl_build_contract_preferences import preferences_path
from src.common.config import LOG_ROOT, OUTPUT_ROOT
from src.common.io_utils import load_json, safe_name, save_json, setup_logging
from src.common.llm_clients import PROVIDERS, default_model

STAGE = "2R_revision_policy_optimization"
OUT_DIR = OUTPUT_ROOT / "classeval" / "rl_revision_policy"
LOG_FILE = LOG_ROOT / "classeval_rl_revision_policy.log"


def policy_path(provider: str, model: str) -> Path:
    return OUT_DIR / safe_name(provider) / safe_name(model) / "policy.json"


def num(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def side_counts(preferences: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for pref in preferences:
        chosen = pref.get("chosen_round")
        rejected = pref.get("rejected_round")
        if chosen:
            counts[f"{chosen}_chosen"] += 1
        if rejected:
            counts[f"{rejected}_rejected"] += 1
    return dict(counts)


def reward_stats(preferences: list[dict[str, Any]]) -> dict[str, float]:
    gaps = [num(pref.get("reward_gap")) for pref in preferences]
    v1 = [num(pref.get("v1_reward")) for pref in preferences]
    v2 = [num(pref.get("v2_reward")) for pref in preferences]

    return {
        "mean_gap": round(mean(gaps), 6) if gaps else 0.0,
        "mean_v1_reward": round(mean(v1), 6) if v1 else 0.0,
        "mean_v2_reward": round(mean(v2), 6) if v2 else 0.0,
    }


def policy_text(counts: dict[str, int], stats: dict[str, float]) -> str:
    v2_wins = counts.get("v2_chosen", 0)
    v1_wins = counts.get("v1_chosen", 0)

    lines = [
        "STAGE 2R REVISION POLICY",
        "",
        "Goal:",
        "Revise a ClassEval contract using task information, the previous contract, and summarized execution feedback.",
        "The output is a revised contract, not code and not tests.",
        "",
        "Learned signal:",
        f"- v2 chosen: {v2_wins}",
        f"- v1 chosen: {v1_wins}",
        f"- mean reward gap: {stats['mean_gap']}",
        f"- mean v1 reward: {stats['mean_v1_reward']}",
        f"- mean v2 reward: {stats['mean_v2_reward']}",
        "",
        "Revision rules:",
        "1. Keep correct clauses from the previous contract.",
        "2. Change the contract only when feedback indicates missing or wrong behavior.",
        "3. Convert failures into general semantic guarantees, not test-specific patches.",
        "4. Prefer concise, executable postconditions over long explanations.",
        "5. Add edge cases only when feedback or docstrings clearly support them.",
        "6. Preserve all method names and signatures exactly.",
        "7. Do not invent exceptions, input restrictions, or runtime checks unless explicitly required.",
        "8. Do not add generic type preconditions that repeat the docstring.",
        "9. If feedback is weak or the previous code passed, keep the previous contract unchanged.",
        "10. Avoid increasing contract length unless it fixes a clear behavioral omission.",
    ]

    if v2_wins <= v1_wins:
        lines += [
            "",
            "Conservative mode:",
            "- Previous contracts often performed as well as or better than revised contracts.",
            "- Revise only failed or clearly incomplete behavior.",
            "- Prefer minimal edits over full rewrites.",
        ]
    else:
        lines += [
            "",
            "Active revision mode:",
            "- Revisions often improved reward.",
            "- Add missing guarantees for failed methods.",
            "- Strengthen ambiguous postconditions when feedback identifies missing behavior.",
        ]

    return "\n".join(lines)


def optimize(preferences: list[dict[str, Any]]) -> dict[str, Any]:
    counts = side_counts(preferences)
    stats = reward_stats(preferences)

    return {
        "stage": STAGE,
        "policy_name": "sequential_feedback_revision_policy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preference_count": len(preferences),
        "round_counts": counts,
        "reward_stats": stats,
        "optimized_policy_text": policy_text(counts, stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a revision policy from v1/v2 contract preferences.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_FILE)

    model = args.model or default_model(args.provider)
    source = preferences_path(args.provider, model)
    target = policy_path(args.provider, model)

    if target.exists() and not args.overwrite:
        print(f"Policy already exists: {target}")
        return

    started = time.perf_counter()
    data = load_json(source)
    preferences = data.get("preferences", [])

    if not isinstance(preferences, list) or not preferences:
        raise ValueError(f"No revision preferences found in {source}")

    policy = optimize(preferences)
    policy.update(
        {
            "provider": args.provider,
            "model": model,
            "source_preferences": str(source),
        }
    )

    save_json(target, policy)

    print("\nClassEval revision policy optimization finished.")
    print(f"Preferences: {len(preferences)}")
    print(f"Output: {target}")
    print(f"Elapsed: {round(time.perf_counter() - started, 2)}s")


if __name__ == "__main__":
    main()