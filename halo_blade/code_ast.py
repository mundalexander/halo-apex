"""Halo Blade — AST-based parsing of Python source files.

Extracts a structural "skeleton": all class and function signatures
without their implementation bodies. Gives an LLM a compact,
token-efficient view of a file's API surface.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Union

AstCallable = Union[ast.FunctionDef, ast.AsyncFunctionDef]


class ASTCodeParser:
    """Parses Python files and extracts structural skeletons."""

    def extract_skeleton(self, file_path: str) -> str:
        """Return the skeleton of ``file_path`` (signatures only)."""
        source = Path(file_path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        blocks: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                blocks.append(self._class_signature(node, indent=0))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                blocks.append(self._function_signature(node, indent=0))

        header = f"# Skeleton: {Path(file_path).name}"
        if not blocks:
            return header + "\n# (no top-level classes or functions)"
        return header + "\n\n" + "\n\n".join(blocks)

    # ------------------------------------------------------------------
    def _class_signature(self, node: ast.ClassDef, indent: int) -> str:
        pad = "    " * indent
        parts: list[str] = []
        for deco in node.decorator_list:
            parts.append(f"{pad}@{ast.unparse(deco)}")
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        header = f"{pad}class {node.name}"
        if bases:
            header += f"({bases})"
        parts.append(header + ":")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append(self._function_signature(child, indent + 1))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def _function_signature(self, node: AstCallable, indent: int) -> str:
        pad = "    " * indent
        parts: list[str] = []
        for deco in node.decorator_list:
            parts.append(f"{pad}@{ast.unparse(deco)}")
        args = ast.unparse(node.args) if node.args else ""
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        parts.append(f"{pad}{keyword} {node.name}({args}){returns}: ...")
        return "\n".join(parts)