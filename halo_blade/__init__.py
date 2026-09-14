"""Halo Blade — agent orchestration, AST parsing & diff execution.

Standalone package: no imports from halo_vector or halo_stream. Vector
results are consumed exclusively through the injected ``VectorSearch``
protocol (structural typing), which halo_vector.VectorEngine satisfies
out of the box.
"""

from .code_ast import ASTCodeParser
from .executor import LocalExecutor
from .file_ops import FileOperations
from .orchestrator import OpenClawOrchestrator, VectorSearch

__all__ = [
    "ASTCodeParser",
    "FileOperations",
    "LocalExecutor",
    "OpenClawOrchestrator",
    "VectorSearch",
]