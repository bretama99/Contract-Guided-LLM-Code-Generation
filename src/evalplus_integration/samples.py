import json
from pathlib import Path
from typing import Any

from src.common.config import DATASETS
from src.common.io_utils import load_json, load_json_list, save_json
from src.common.task_utils import select_tasks
from src.evalplus_integration.paths import SUPPORTED_METHODS, generation_folder, samples_folder


FAILURE_MESSAGE = "Missing or failed generation counted as failure"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_generations(folder: Path, suffix: str) -> dict[str, dict[str, Any]]:
    if not folder.exists():
        return {}

    records = {}
    for path in sorted(folder.glob(f"*{suffix}")):
        data = load_json(path)
        if isinstance(data, dict) and data.get("task_id"):
            records[str(data["task_id"])] = {"path": str(path), "record": data}
    return records


def failure_solution(entry_point: str | None) -> str:
    name = (entry_point or "candidate").strip() or "candidate"
    return f"def {name}(*args, **kwargs):\n    raise NotImplementedError({FAILURE_MESSAGE!r})\n"


def generation_issue(info: dict[str, Any] | None) -> str | None:
    if info is None:
        return "missing_generation"

    record = info["record"]
    if record.get("status") != "success":
        return "generation_failed"

    code = record.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        return "empty_generated_code"

    return None


def build_sample(task: dict[str, Any], gen_info: dict[str, Any] | None) -> tuple[dict[str, str], str | None]:
    issue = generation_issue(gen_info)

    if issue:
        return {
            "task_id": str(task["task_id"]),
            "solution": failure_solution(task.get("entry_point")),
        }, issue

    return {
        "task_id": str(task["task_id"]),
        "solution": gen_info["record"]["generated_code"],
    }, None


def export_evalplus_samples(
    *,
    dataset: str,
    method: str,
    provider: str,
    model: str,
    start: int = 0,
    count: int | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    if dataset not in DATASETS or "evalplus_dataset" not in DATASETS[dataset]:
        raise ValueError(f"Dataset is not an EvalPlus dataset: {dataset}")

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported EvalPlus method: {method}")

    info = DATASETS[dataset]
    tasks = select_tasks(load_json_list(info["path"]), start=start, count=count)

    method_info = SUPPORTED_METHODS[method]
    gen_dir = generation_folder(method, dataset, provider, model)
    generations = load_generations(gen_dir, method_info["suffix"])

    successful_rows = []
    complete_rows = []
    skipped = []

    for task in tasks:
        task_id = str(task["task_id"])
        row, issue = build_sample(task, generations.get(task_id))

        complete_rows.append(row)

        if issue:
            skipped.append({"task_id": task_id, "reason": issue})
        else:
            successful_rows.append(row)

    skipped_by_reason = {}
    for item in skipped:
        skipped_by_reason[item["reason"]] = skipped_by_reason.get(item["reason"], 0) + 1

    out_dir = samples_folder(method, dataset, provider, model)
    successful_path = out_dir / "samples.jsonl"
    complete_path = out_dir / "samples_complete.jsonl"
    summary_path = out_dir / "export_summary.json"

    summary = {
        "dataset": dataset,
        "evalplus_dataset": info["evalplus_dataset"],
        "method": method,
        "provider": provider,
        "model": model,
        "task_count": len(tasks),
        "successful_sample_count": len(successful_rows),
        "complete_sample_count": len(complete_rows),
        "filled_failure_count": len(skipped),
        "skipped_by_reason": skipped_by_reason,
        "generation_folder": str(gen_dir),
        "skipped": skipped,
    }

    write_jsonl(successful_path, successful_rows)
    write_jsonl(complete_path, complete_rows)
    save_json(summary_path, summary)

    return successful_path, complete_path, summary_path, summary