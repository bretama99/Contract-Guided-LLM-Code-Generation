from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "humaneval": {
        "label": "HumanEval",
        "path": ROOT / "data" / "processed" / "humaneval" / "humaneval_tasks.json",
    },
    "bigcodebench": {
        "label": "BigCodeBench",
        "path": ROOT / "data" / "processed" / "bigcodebench" / "bigcodebench_tasks.json",
    },
    "evalplus": {
        "label": "EvalPlus-HumanEval+",
        "path": ROOT / "data" / "processed" / "evalplus" / "humanevalplus_tasks.json",
        "evalplus_dataset": "humaneval",
    },
    "evalplus_mbpp": {
        "label": "EvalPlus-MBPP+",
        "path": ROOT / "data" / "processed" / "evalplus" / "mbppplus_tasks.json",
        "evalplus_dataset": "mbpp",
    },
    "livecodebench": {
        "label": "LiveCodeBench",
        "path": ROOT / "data" / "processed" / "livecodebench" / "livecodebench_tasks.json",
    },
}

OUTPUT_ROOT = ROOT / "outputs"
RESULTS_ROOT = ROOT / "results"
LOG_ROOT = ROOT / "logs"

RAW_CONTRACT_ROOT = OUTPUT_ROOT / "contracts" / "raw"
RAW_CONTRACT_CODE_ROOT = OUTPUT_ROOT / "contract_guided_generation" / "raw_contracts"
RAW_CONTRACT_RESULTS_ROOT = RESULTS_ROOT / "contract_guided_generation" / "raw_contracts"