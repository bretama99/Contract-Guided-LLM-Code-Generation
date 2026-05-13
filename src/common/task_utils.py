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


def task_identifier(task: dict[str, Any]) -> str:
    for key in TASK_ID_FIELDS:
        value = clean(task.get(key))
        if value:
            return value
    raise KeyError("Task is missing task_id/id/problem_id.")


def task_entry_point(task: dict[str, Any]) -> str:
    for key in ENTRY_POINT_FIELDS:
        value = clean(task.get(key))
        if value:
            return value
    return ""


def task_prompt(task: dict[str, Any]) -> str:
    candidates = [clean(task.get(key)) for key in PROMPT_FIELDS if clean(task.get(key))]
    return max(candidates, key=len) if candidates else ""


def select_tasks(
    tasks: list[dict[str, Any]],
    start: int = 0,
    count: int | None = None,
) -> list[dict[str, Any]]:
    if count is None:
        return tasks[start:]
    return tasks[start : start + count]