from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Any
from src.common.io_utils import safe_name
from src.common.parsing import extract_python_code
JsonDict = dict[str, Any]

def text(value: Any) -> str:
    return "" if value is None else str(value).strip()

def normalize_code_text(code: str) -> str:
    return code.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "„": '"',
                "‟": '"',
                "‘": "'",
                "’": "'",
                "‚": "'",
                "‛": "'",
                "\u00a0": " ",
            }
        )
    )
    
def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []

def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}

def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)

def task_id(task: JsonDict) -> str:
    value = text(task.get("task_id"))
    if not value:
        raise ValueError("ClassEval task is missing task_id.")
    return value

def class_name(task: JsonDict) -> str:
    value = text(task.get("class_name"))
    if not value:
        raise ValueError(f"{task_id(task)} is missing class_name.")
    return value

def skeleton(task: JsonDict) -> str:
    value = text(task.get("skeleton"))
    if not value:
        raise ValueError(f"{task_id(task)} is missing skeleton.")
    return value

def output_path(base_dir: Path, task: JsonDict, provider: str, model: str, suffix: str) -> Path:
    return (
        base_dir
        / safe_name(provider)
        / safe_name(model)
        / f"{safe_name(task_id(task))}_{suffix}.json"
    )

def read_template(path: Path, required: tuple[str, ...]) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    template = path.read_text(encoding="utf-8")
    missing = [f"{{{name}}}" for name in required if f"{{{name}}}" not in template]
    if missing:
        raise ValueError(f"{path} missing placeholders: {', '.join(missing)}")
    return template

def fill_template(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace(f"{{{key}}}", value)
    return template

def ensure_no_reference_leak(task: JsonDict, prompt: str) -> None:
    forbidden = [task.get("solution_code"), task.get("test")]
    for method in as_list(task.get("methods_info")):
        if isinstance(method, dict):
            forbidden.extend(
                value
                for key, value in method.items()
                if "solution" in key.lower() or "test" in key.lower()
            )

    for value in forbidden:
        leaked = text(value)
        if len(leaked) >= 40 and leaked in prompt:
            raise ValueError(f"Reference/test content leaked into prompt for {task_id(task)}.")
def parse_module(code: str) -> ast.Module:
    try:
        return ast.parse(normalize_code_text(code))
    except SyntaxError as exc:
        raise SyntaxError(f"Invalid Python source: {exc}") from exc
def has_incomplete_method_body(node: ast.FunctionDef) -> bool:
    body = [
        stmt
        for stmt in node.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]

    if not body:
        return True

    return all(_is_placeholder_statement(stmt) for stmt in body)


def _is_placeholder_statement(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True

    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    ):
        return True

    if isinstance(stmt, ast.Raise):
        exc = stmt.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"

    return False

def find_class(tree: ast.Module, expected: str) -> ast.ClassDef | None:
    return next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == expected),
        None,
    )

def is_staticmethod(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(dec, ast.Name) and dec.id == "staticmethod"
        or isinstance(dec, ast.Attribute) and dec.attr == "staticmethod"
        for dec in node.decorator_list
    )

def method_profiles(code: str, expected_class: str) -> dict[str, tuple[str, bool]]:
    cls = find_class(parse_module(code), expected_class)
    if cls is None:
        return {}
    return {
        node.name: (ast.dump(node.args, include_attributes=False), is_staticmethod(node))
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }

def contains_tests(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "unittest" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "unittest":
            return True
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id.endswith("TestCase"):
                    return True
                if isinstance(base, ast.Attribute) and base.attr == "TestCase":
                    return True
    return False

def verify_class_code(code: str, task: JsonDict, *, check_methods: bool = True) -> None:
    expected = class_name(task)
    tree = parse_module(code)
    if find_class(tree, expected) is None:
        raise ValueError(f"Generated code does not contain class {expected}.")
    if contains_tests(tree):
        raise ValueError("Generated code appears to contain tests.")
    if not check_methods:
        return

    required = method_profiles(skeleton(task), expected)
    generated = method_profiles(code, expected)
    generated_class = find_class(tree, expected)
    if generated_class is not None:
        incomplete = sorted(
            node.name
            for node in generated_class.body
            if isinstance(node, ast.FunctionDef) and has_incomplete_method_body(node)
        )
    if incomplete:
        raise ValueError(f"Incomplete method bodies: {incomplete}")
    missing = sorted(set(required) - set(generated))
    changed = sorted(name for name, profile in required.items() if generated.get(name) != profile)
    if missing:
        raise ValueError(f"Missing methods: {missing}")
    if changed:
        raise ValueError(f"Changed method signatures or staticmethod decorators: {changed}")

def extract_class_code(raw_response: str, task: JsonDict, *, check_methods: bool = True) -> str:
    raw_response = normalize_code_text(raw_response)
    code = extract_python_code(raw_response, entry_point=None, validate=False).strip()
    code = normalize_code_text(code)
    if not code:
        raise ValueError("Generated code is empty.")
    verify_class_code(code, task, check_methods=check_methods)
    return code

def safe_methods_info(task: JsonDict) -> list[JsonDict]:
    result: list[JsonDict] = []
    for method in as_list(task.get("methods_info")):
        if isinstance(method, dict):
            result.append(
                {
                    key: value
                    for key, value in method.items()
                    if "test" not in key.lower() and "solution" not in key.lower()
                }
            )
    return result
__all__ = [
    "JsonDict",
    "as_dict",
    "as_list",
    "class_name",
    "compact_json",
    "ensure_no_reference_leak",
    "extract_class_code",
    "fill_template",
    "has_incomplete_method_body",
    "method_profiles",
    "normalize_code_text",
    "output_path",
    "pretty_json",
    "read_template",
    "safe_methods_info",
    "skeleton",
    "task_id",
    "text",
    "verify_class_code",
]