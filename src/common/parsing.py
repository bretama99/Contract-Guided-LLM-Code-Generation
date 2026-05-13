import ast
import json
import re
from typing import Any


_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

_CODE_START_RE = re.compile(
    r"(?m)^(from\s+\S+\s+import\s+.*|import\s+.*|def\s+\w+\s*\(|class\s+\w+\s*)"
)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()

    block = _JSON_BLOCK_RE.search(cleaned)
    if block:
        cleaned = block.group(1).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid JSON object found in model response.")
        value = json.loads(cleaned[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got {type(value).__name__}")

    return value


def extract_python_code(text: str, entry_point: str | None = None, validate: bool = True) -> str:
    cleaned = (text or "").strip()

    block = _CODE_BLOCK_RE.search(cleaned)
    if block:
        cleaned = block.group(1).strip()

    start = _CODE_START_RE.search(cleaned)
    if start:
        cleaned = cleaned[start.start() :].strip()

    if validate:
        tree = ast.parse(cleaned)

        if entry_point:
            function_names = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }

            if entry_point not in function_names:
                raise ValueError(f"Missing required entry point: {entry_point}")

    return cleaned

import ast


COMMON_ALIAS_IMPORTS = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "sns": "import seaborn as sns",
}


class _NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.defined: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Param)):
            self.defined.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        for arg in node.args.args + node.args.kwonlyargs:
            self.defined.add(arg.arg)
        if node.args.vararg:
            self.defined.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)


def add_missing_common_alias_imports(code: str) -> str:
    tree = ast.parse(code)

    collector = _NameCollector()
    collector.visit(tree)

    missing_imports = []
    for alias, import_line in COMMON_ALIAS_IMPORTS.items():
        if alias in collector.loaded and alias not in collector.defined:
            missing_imports.append(import_line)

    if not missing_imports:
        return code

    return "\n".join(missing_imports) + "\n\n" + code