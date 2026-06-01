from __future__ import annotations

from pathlib import Path

from src.common.config import OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import safe_name
from src.common.raw_contract_paths import raw_contract_code_folder

EVALPLUS_SAMPLE_ROOT = OUTPUT_ROOT / "evalplus_samples"
EVALPLUS_RESULT_ROOT = RESULTS_ROOT / "evalplus"

SUPPORTED_METHODS = {
    "vanilla": {"suffix": "_vanilla.json"},
    "raw_contracts": {"suffix": "_raw_contract_guided.json"},
    "optimized_rl": {"suffix": "_optimized_contract_guided.json"},
}


def generation_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    if method == "vanilla":
        return OUTPUT_ROOT / "vanilla" / safe_name(provider) / safe_name(model) / safe_name(dataset)

    if method == "raw_contracts":
        return raw_contract_code_folder(dataset, provider, model)

    if method == "optimized_rl":
        from src.rl_method_level.rl_generate_contract import optimized_contract_code_folder
        return optimized_contract_code_folder(dataset, provider, model)

    raise ValueError(f"Unsupported EvalPlus method: {method}")
def samples_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported EvalPlus method: {method}")

    return EVALPLUS_SAMPLE_ROOT / safe_name(method) / safe_name(provider) / safe_name(model) / safe_name(dataset)


def results_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported EvalPlus method: {method}")

    return EVALPLUS_RESULT_ROOT / safe_name(method) / safe_name(provider) / safe_name(model) / safe_name(dataset)