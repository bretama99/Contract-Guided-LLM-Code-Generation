from pathlib import Path

from src.common.config import OUTPUT_ROOT, RESULTS_ROOT
from src.common.io_utils import safe_name
from src.common.raw_contract_paths import raw_contract_code_folder


EVALPLUS_SAMPLE_ROOT = OUTPUT_ROOT / "evalplus_samples"
EVALPLUS_RESULT_ROOT = RESULTS_ROOT / "evalplus"


SUPPORTED_METHODS = {
    "vanilla": {
        "suffix": "_vanilla.json",
    },
    "raw_contracts": {
        "suffix": "_raw_contract_guided.json",
    },
}


def generation_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    if method == "vanilla":
        return OUTPUT_ROOT / "vanilla" / safe_name(provider) / safe_name(model) / dataset

    if method == "raw_contracts":
        return raw_contract_code_folder(dataset, provider, model)

    raise ValueError(f"Unsupported EvalPlus method: {method}")


def samples_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    return EVALPLUS_SAMPLE_ROOT / method / safe_name(provider) / safe_name(model) / dataset


def results_folder(method: str, dataset: str, provider: str, model: str) -> Path:
    return EVALPLUS_RESULT_ROOT / method / safe_name(provider) / safe_name(model) / dataset