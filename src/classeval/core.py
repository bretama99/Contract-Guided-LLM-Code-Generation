from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.common.io_utils import safe_name
from src.common.parsing import extract_python_code

JsonDict = dict[str, Any]

REFERENCE_LEAK_SIMILARITY_THRESHOLD = 0.72
REFERENCE_CODE_SIMILARITY_WARNING_THRESHOLD = 0.85
MIN_REFERENCE_CHARS = 80


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


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


def task_id(task: JsonDict) -> str:
    value = text(task.get("task_id"))
    if not value:
        raise ValueError("ClassEval task is missing task_id.")
    return value


def class_name(task: JsonDict) -> str:
    value = text(task.get("class_name")) or text(task.get("entry_point"))
    if not value:
        raise ValueError(f"{task_id(task)} is missing class_name.")
    return value


def skeleton(task: JsonDict) -> str:
    value = text(task.get("skeleton"))
    if not value:
        raise ValueError(f"{task_id(task)} is missing skeleton.")
    return value


def output_path(base_dir: Path, task: JsonDict, provider: str, model: str, suffix: str) -> Path:
    return base_dir / safe_name(provider) / safe_name(model) / f"{safe_name(task_id(task))}_{suffix}.json"


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


def safe_methods_info(task: JsonDict) -> list[JsonDict]:
    blocked = ("test", "solution", "reference", "answer", "expected", "oracle", "ground_truth")
    return [
        {key: value for key, value in method.items() if not any(item in key.lower() for item in blocked)}
        for method in as_list(task.get("methods_info"))
        if isinstance(method, dict)
    ]


def _looks_reference_key(key: str, *, include_tests: bool) -> bool:
    lowered = key.lower()
    reference_terms = ("solution", "reference", "canonical", "ground_truth", "oracle")
    test_terms = ("test", "tests", "unit_test", "answer", "expected")

    return any(term in lowered for term in reference_terms) or (
        include_tests and any(term in lowered for term in test_terms)
    )


def reference_blobs(task: JsonDict, *, include_tests: bool = True) -> list[tuple[str, str]]:
    blobs: list[tuple[str, str]] = []

    for key, value in task.items():
        if _looks_reference_key(str(key), include_tests=include_tests) and text(value):
            blobs.append((str(key), text(value)))

    for index, method in enumerate(as_list(task.get("methods_info"))):
        if not isinstance(method, dict):
            continue

        for key, value in method.items():
            if _looks_reference_key(str(key), include_tests=include_tests) and text(value):
                blobs.append((f"methods_info[{index}].{key}", text(value)))

    return blobs


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", normalize_code_text(value))


def _tokens(value: str) -> list[str]:
    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\],.:]",
        normalize_code_text(value).lower(),
    )


def _token_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)

    if len(left_tokens) < 20 or len(right_tokens) < 20:
        return 0.0

    return SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens), autojunk=True).ratio()


def reference_leakage_report(
    task: JsonDict,
    candidate_text: str,
    *,
    include_tests: bool = False,
    similarity_threshold: float = REFERENCE_CODE_SIMILARITY_WARNING_THRESHOLD,
) -> JsonDict:
    candidate = text(candidate_text)
    compact_candidate = _compact(candidate)

    best: JsonDict = {
        "warning": False,
        "matched_source": None,
        "exact_substring_match": False,
        "similarity": 0.0,
        "threshold": similarity_threshold,
    }

    if not candidate:
        return best

    for source, reference in reference_blobs(task, include_tests=include_tests):
        if len(reference) < MIN_REFERENCE_CHARS:
            continue

        exact = _compact(reference) in compact_candidate
        similarity = _token_similarity(candidate, reference)

        if exact or similarity > float(best["similarity"]):
            best.update(
                {
                    "warning": bool(exact or similarity >= similarity_threshold),
                    "matched_source": source,
                    "exact_substring_match": exact,
                    "similarity": round(similarity, 6),
                }
            )

        if exact:
            break

    return best


def ensure_no_reference_leak(task: JsonDict, prompt: str) -> None:
    report = reference_leakage_report(
        task,
        prompt,
        include_tests=True,
        similarity_threshold=REFERENCE_LEAK_SIMILARITY_THRESHOLD,
    )

    if report.get("warning"):
        raise ValueError(
            f"Reference/test content leaked into prompt for {task_id(task)}: "
            f"source={report.get('matched_source')} "
            f"similarity={report.get('similarity')} "
            f"exact={report.get('exact_substring_match')}"
        )


def parse_module(code: str) -> ast.Module:
    try:
        return ast.parse(normalize_code_text(code))
    except SyntaxError as exc:
        raise SyntaxError(f"Invalid Python source: {exc}") from exc


def find_class(tree: ast.Module, expected: str) -> ast.ClassDef | None:
    return next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == expected), None)


def is_staticmethod(node: ast.FunctionDef) -> bool:
    return any(
        (isinstance(dec, ast.Name) and dec.id == "staticmethod")
        or (isinstance(dec, ast.Attribute) and dec.attr == "staticmethod")
        for dec in node.decorator_list
    )


def _is_placeholder(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True

    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return True

    if isinstance(stmt, ast.Raise):
        exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"

    return False


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

    return not body or all(_is_placeholder(stmt) for stmt in body)


def method_profiles(code: str, expected_class: str) -> dict[str, tuple[str, bool]]:
    cls = find_class(parse_module(code), expected_class)
    if cls is None:
        return {}

    return {
        node.name: (
            ast.dump(node.args, include_attributes=False)
            + "|returns="
            + (ast.dump(node.returns, include_attributes=False) if node.returns is not None else "None"),
            is_staticmethod(node),
        )
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


def _allowed_top_level(node: ast.stmt, expected: str) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True

    if isinstance(node, ast.ClassDef):
        return node.name == expected

    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _metadata_method_names(task: JsonDict) -> set[str]:
    return {
        text(method.get("method_name"))
        for method in safe_methods_info(task)
        if text(method.get("method_name"))
    }


def verify_class_code(code: str, task: JsonDict, *, check_methods: bool = True) -> None:
    expected = class_name(task)
    tree = parse_module(code)
    generated_class = find_class(tree, expected)

    if generated_class is None:
        raise ValueError(f"Generated code does not contain class {expected}.")

    if contains_tests(tree):
        raise ValueError("Generated code appears to contain tests.")

    bad_top_level = [type(node).__name__ for node in tree.body if not _allowed_top_level(node, expected)]
    if bad_top_level:
        raise ValueError(f"Generated code contains disallowed top-level code: {bad_top_level}")

    if not check_methods:
        return

    try:
        required = method_profiles(skeleton(task), expected)
    except SyntaxError:
        required = {}

    generated = method_profiles(code, expected)
    required_names = set(required) or _metadata_method_names(task)

    incomplete = sorted(
        node.name
        for node in generated_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name != "__init__"
        and has_incomplete_method_body(node)
    )
    missing = sorted(required_names - set(generated))
    changed = sorted(name for name, profile in required.items() if generated.get(name) != profile)

    if incomplete:
        raise ValueError(f"Incomplete method bodies: {incomplete}")

    if missing:
        raise ValueError(f"Missing methods: {missing}")

    if changed:
        raise ValueError(f"Changed method signatures, return annotations, or staticmethod decorators: {changed}")


def extract_class_code(raw_response: str, task: JsonDict, *, check_methods: bool = True) -> str:
    code = extract_python_code(
        normalize_code_text(raw_response),
        entry_point=None,
        validate=False,
    ).strip()

    if not code:
        raise ValueError("Generated code is empty.")

    code = normalize_code_text(code)
    verify_class_code(code, task, check_methods=check_methods)
    return code


__all__ = [
    "JsonDict",
    "as_dict",
    "as_list",
    "class_name",
    "compact_json",
    "ensure_no_reference_leak",
    "extract_class_code",
    "fill_template",
    "find_class",
    "has_incomplete_method_body",
    "method_profiles",
    "normalize_code_text",
    "output_path",
    "parse_module",
    "pretty_json",
    "read_template",
    "reference_blobs",
    "reference_leakage_report",
    "safe_methods_info",
    "skeleton",
    "task_id",
    "text",
    "verify_class_code",
]