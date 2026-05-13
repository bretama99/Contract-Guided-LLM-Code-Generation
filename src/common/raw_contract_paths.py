from pathlib import Path

from src.common.config import (
    RAW_CONTRACT_CODE_ROOT,
    RAW_CONTRACT_RESULTS_ROOT,
    RAW_CONTRACT_ROOT,
)
from src.common.io_utils import safe_name


def raw_contract_folder(dataset: str, provider: str, model: str) -> Path:
    return RAW_CONTRACT_ROOT / safe_name(provider) / safe_name(model) / dataset


def raw_contract_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return raw_contract_folder(dataset, provider, model) / f"{safe_name(task_id)}_contract.json"


def raw_contract_code_folder(dataset: str, provider: str, model: str) -> Path:
    return RAW_CONTRACT_CODE_ROOT / safe_name(provider) / safe_name(model) / dataset


def raw_contract_code_path(dataset: str, task_id: str, provider: str, model: str) -> Path:
    return (
        raw_contract_code_folder(dataset, provider, model)
        / f"{safe_name(task_id)}_raw_contract_guided.json"
    )


def raw_contract_results_folder(dataset: str, provider: str, model: str) -> Path:
    return RAW_CONTRACT_RESULTS_ROOT / safe_name(provider) / safe_name(model) / dataset