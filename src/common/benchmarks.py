from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.common.config import DATASETS
from src.common.execution import (
    evaluate_bigcodebench_candidate,
    evaluate_humaneval_candidate,
    evaluate_livecodebench_candidate,
)


EvaluationResult = dict[str, Any]
Evaluator = Callable[[dict[str, Any], str, float], EvaluationResult]


@dataclass(frozen=True)
class Benchmark:
    key: str
    label: str
    path: Path
    evaluator: Evaluator | None = None
    evalplus_dataset: str | None = None

    @property
    def is_evalplus(self) -> bool:
        return self.evalplus_dataset is not None


def _registry() -> dict[str, Benchmark]:
    evaluators: dict[str, Evaluator] = {
        "humaneval": evaluate_humaneval_candidate,
        "bigcodebench": evaluate_bigcodebench_candidate,
        "livecodebench": evaluate_livecodebench_candidate,

    }

    return {
        key: Benchmark(
            key=key,
            label=info["label"],
            path=Path(info["path"]),
            evaluator=evaluators.get(key),
            evalplus_dataset=info.get("evalplus_dataset"),
        )
        for key, info in DATASETS.items()
    }


BENCHMARKS = _registry()


def benchmark_keys() -> list[str]:
    return sorted(BENCHMARKS)


def evalplus_keys() -> list[str]:
    return sorted(key for key, bench in BENCHMARKS.items() if bench.is_evalplus)


def get_benchmark(key: str) -> Benchmark:
    try:
        return BENCHMARKS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {key}") from exc


def require_native_evaluator(benchmark: Benchmark, method: str) -> None:
    if benchmark.is_evalplus:
        raise SystemExit(
            f"EvalPlus datasets must be evaluated with: "
            f"python scripts/evaluate_evalplus.py --method {method}"
        )

    if benchmark.evaluator is None:
        raise NotImplementedError(
            f"No native evaluator registered for dataset '{benchmark.key}'. "
            f"Add tests and register an evaluator in src/common/benchmarks.py."
        )


def evaluate_candidate(
    *,
    benchmark: Benchmark,
    task: dict[str, Any],
    code: str,
    timeout: float,
) -> EvaluationResult:
    require_native_evaluator(benchmark, method="native")

    assert benchmark.evaluator is not None
    return benchmark.evaluator(task, code, timeout)