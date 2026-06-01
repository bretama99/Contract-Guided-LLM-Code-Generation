from __future__ import annotations

from typing import Any

TASK_ID_FIELDS = ("task_id", "id", "problem_id", "question_id")
ENTRY_POINT_FIELDS = ("entry_point", "entrypoint", "function_name", "name")
PROMPT_FIELDS = (
    "complete_prompt",
    "prompt",
    "instruct_prompt",
    "code_prompt",
    "instruction",
    "description",
    "task_description",
    "problem_statement",
)


def clean(value: Any) -> str:
    return str(value or "").strip()


def first_present(task: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = clean(task.get(field))
        if value:
            return value
    return ""


def task_identifier(task: dict[str, Any]) -> str:
    value = first_present(task, TASK_ID_FIELDS)
    if not value:
        raise KeyError("Task is missing task_id/id/problem_id/question_id.")
    return value


def task_entry_point(task: dict[str, Any]) -> str:
    return first_present(task, ENTRY_POINT_FIELDS)


def task_prompt(task: dict[str, Any]) -> str:
    candidates = [clean(task.get(field)) for field in PROMPT_FIELDS if clean(task.get(field))]
    return max(candidates, key=len) if candidates else ""


def select_tasks(tasks: list[dict[str, Any]], start: int = 0, count: int | None = None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be non-negative")
    if count is not None and count < 0:
        raise ValueError("--count must be non-negative")
    return tasks[start:] if count is None else tasks[start : start + count]