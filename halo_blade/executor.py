"""Halo Blade — local shell command execution with log truncation."""

from __future__ import annotations

import subprocess
from typing import Any


class LocalExecutor:
    """Runs shell commands locally and truncates long log output."""

    def __init__(self, max_log_lines: int = 50, timeout: int = 120) -> None:
        self.max_log_lines = max_log_lines
        self.timeout = timeout

    def run_command(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        """Execute ``command`` in a shell and return a structured result."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "command": command,
                "returncode": result.returncode,
                "stdout": self._truncate(result.stdout),
                "stderr": self._truncate(result.stderr),
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            return {
                "command": command,
                "returncode": -1,
                "stdout": self._truncate(stdout),
                "stderr": f"Timeout after {self.timeout}s.",
            }

    def _truncate(self, text: str) -> str:
        """Cut ``text`` down to at most ``max_log_lines`` lines."""
        if not text:
            return ""
        lines = text.splitlines()
        if len(lines) <= self.max_log_lines:
            return text
        kept = lines[: self.max_log_lines]
        return (
            "\n".join(kept)
            + f"\n... [{len(lines) - self.max_log_lines} lines truncated]"
        )