"""Halo Blade — file operations: line-range reading and unified diffs."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FileOperations:
    """Safe, minimal file operations used by the Halo Blade engine."""

    def read_line_range(self, file_path: str, start: int, end: int) -> str:
        """Read lines ``start..end`` (1-based, inclusive) from a file."""
        if start < 1:
            raise ValueError("start must be >= 1")
        if start > end:
            raise ValueError("start must be <= end")
        path = Path(file_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[start - 1 : end])

    def read_file(self, file_path: str) -> str:
        """Read an entire file."""
        return Path(file_path).read_text(encoding="utf-8")

    def apply_patch(self, diff_path: str, target_dir: str = ".") -> str:
        """Apply a unified diff with ``git apply``.

        Escalates leniency on failure:
        1. strict ``git apply``
        2. ``--recount`` — recomputes hunk line counts (tolerates
           counting mistakes in LLM-generated diffs)
        3. ``--unidiff-zero`` — accepts zero-context hunks, which
           ``git apply`` rejects by default (LLMs sometimes emit
           context-free diffs)

        Raises ``RuntimeError`` if all attempts fail.
        """
        attempts = (
            ["git", "apply", "--verbose"],
            ["git", "apply", "--recount", "--verbose"],
            ["git", "apply", "--unidiff-zero", "--recount", "--verbose"],
        )
        last_error = ""
        for cmd in attempts:
            result = subprocess.run(
                cmd + [diff_path],
                cwd=target_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return "Patch applied successfully."
            last_error = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"git apply failed:\n{last_error}")