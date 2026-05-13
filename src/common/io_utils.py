import json
import logging
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty JSON file: {path}")

    return json.loads(text)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    cleaned = cleaned.strip("._-")
    return cleaned or "unnamed"


def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def require_file(path: Path, description: str = "file") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")