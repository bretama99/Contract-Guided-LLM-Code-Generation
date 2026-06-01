from __future__ import annotations

import ast
import json
import re
from typing import Any

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.I | re.S)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.I | re.S)
_CODE_START_RE = re.compile(
    r"(?m)^(from\s+\S+\s+import\s+.*|import\s+.*|def\s+\w+\s*\(|class\s+\w+\s*)"
)

COMMON_ALIAS_IMPORTS = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "sns": "import seaborn as sns",
}


def extract_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    block = _JSON_BLOCK_RE.search(content)
    if block:
        content = block.group(1).strip()

    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No valid JSON object found in model response.")
        value = json.loads(content[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object, got {type(value).__name__}")

    return value


def extract_python_code(text: str, entry_point: str | None = None, validate: bool = True) -> str:
    code = (text or "").strip()

    block = _CODE_BLOCK_RE.search(code)
    if block:
        code = block.group(1).strip()

    start = _CODE_START_RE.search(code)
    if start:
        code = code[start.start() :].strip()

    if validate:
        tree = ast.parse(code)
        if entry_point and not _has_entry_point(tree, entry_point):
            raise ValueError(f"Missing required entry point: {entry_point}")

    return code


def _has_entry_point(tree: ast.Module, entry_point: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == entry_point
        for node in tree.body
    )


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
        for arg in [*node.args.args, *node.args.kwonlyargs]:
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
            self.defined.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)


def add_missing_common_alias_imports(code: str) -> str:
    tree = ast.parse(code)
    collector = _NameCollector()
    collector.visit(tree)

    imports = [
        import_line
        for alias, import_line in COMMON_ALIAS_IMPORTS.items()
        if alias in collector.loaded and alias not in collector.defined
    ]

    return "\n".join(imports) + "\n\n" + code if imports else code