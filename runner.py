"""Halo Apex production runner — task execution with full safety ring.

Production workflow around the standalone subbrands (halo_blade +
halo_vector):

1. Auto-index the target file into the persistent vector store.
2. Create a dedicated git branch (``halo/<task-slug>``) — the current
   branch is never patched directly.
3. Ask the LLM for a unified diff; failed applications are fed back to
   the LLM and retried (``HALO_TASK_MAX_RETRIES``, default 3).
4. Verification gate after every patch: ``py_compile`` (always) and
   ``pytest`` (if the target repo has a ``tests/`` directory). Any
   failure triggers an automatic rollback of the target file.
5. On success: commit the patched file on the task branch, re-index it
   into the vector store and report branch + diffstat for review.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from config import halo_settings
from halo_blade import FileOperations, LocalExecutor, OpenClawOrchestrator
from halo_stream import HaloStreamProvider
from halo_vector import EmbeddingService, VectorEngine

BRANCH_PREFIX = "halo/"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command inside ``repo`` and return the completed process."""
    result = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}".strip()
        )
    return result


def _slugify(text: str, max_length: int = 24) -> str:
    """Turn a task description into a short branch-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "task"


class TaskRunner:
    """Production-safe single-file task execution."""

    def __init__(self) -> None:
        self.settings = halo_settings
        persist_dir = Path(self.settings.halo_vector_persist_dir).expanduser()
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.embedding_service = EmbeddingService(
            base_url=self.settings.halo_vector_base_url,
            model_name=self.settings.halo_vector_embedding_model,
        )
        self.vector_engine = VectorEngine(
            persist_dir=str(persist_dir),
            embedding_service=self.embedding_service,
        )
        self.executor = LocalExecutor(max_log_lines=self.settings.halo_max_log_lines)
        self.file_ops = FileOperations()

    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Check all active subsystems and return a status report."""
        report: dict[str, Any] = {}

        provider = HaloStreamProvider(
            enabled=self.settings.use_halo_stream,
            backend_url=self.settings.halo_stream_backend_url,
            default_url=self.settings.halo_blade_llm_url,
        )
        report["stream"] = provider.stream_status()

        vector_ok = False
        try:
            requests.get(
                self.settings.halo_vector_base_url.rstrip("/") + "/models", timeout=5
            ).raise_for_status()
            vector = self.embedding_service.embed_text("health check")
            vector_ok = True
            report["vector"] = {
                "ok": True,
                "model": self.settings.halo_vector_embedding_model,
                "dim": len(vector),
            }
        except Exception as exc:
            report["vector"] = {"ok": False, "error": str(exc)}

        blade_ok = False
        blade_url = provider.get_backend_url(self.settings.halo_blade_llm_url)
        try:
            requests.get(
                blade_url.rstrip("/") + "/models", timeout=5
            ).raise_for_status()
            blade_ok = True
            report["blade"] = {"ok": True, "url": blade_url}
        except Exception as exc:
            report["blade"] = {"ok": False, "url": blade_url, "error": str(exc)}

        report["persist_dir"] = str(self.persist_dir)
        report["indexed_documents"] = self.vector_engine.size
        report["healthy"] = vector_ok and blade_ok
        return report

    # ------------------------------------------------------------------
    def index_path(self, path: Path) -> int:
        """Index a single .py file or a whole directory tree."""
        if path.is_file():
            files = [path]
        else:
            files = [
                p
                for p in sorted(path.rglob("*.py"))
                if "venv" not in p.parts
                and "__pycache__" not in p.parts
                and not any(part.startswith(".") for part in p.parts)
            ]
        count = 0
        for f in files:
            text = f.read_text(encoding="utf-8")
            if text.strip():
                self.vector_engine.index_file(str(f), text)
                count += 1
        return count

    # ------------------------------------------------------------------
    def run_task(self, task: str, file_path: Path) -> dict[str, Any]:
        """Run ``task`` on ``file_path`` with the full production safety ring."""
        started = time.time()
        file_path = file_path.resolve()
        repo_root = Path(
            _git(file_path.parent, "rev-parse", "--show-toplevel").stdout.strip()
        )
        rel = file_path.relative_to(repo_root)

        # --- Safety: target file must be tracked and clean -------------
        if (
            _git(
                repo_root, "ls-files", "--error-unmatch", str(rel), check=False
            ).returncode
            != 0
        ):
            raise RuntimeError(f"Target file is not tracked by git: {rel}")
        if _git(repo_root, "status", "--porcelain", "--", str(rel)).stdout.strip():
            raise RuntimeError(
                f"Target file has uncommitted changes: {rel} — commit or stash first."
            )

        # --- Branch per task --------------------------------------------
        original_branch = _git(
            repo_root, "rev-parse", "--abbrev-ref", "HEAD"
        ).stdout.strip()
        branch = self._unique_branch(repo_root, _slugify(task))
        _git(repo_root, "checkout", "-b", branch)

        # --- Fresh retrieval context -------------------------------------
        self.index_path(file_path)

        orchestrator = OpenClawOrchestrator(
            llm_base_url=self.settings.halo_blade_llm_url,
            llm_model=self.settings.halo_blade_llm_model,
            vector_engine=self.vector_engine,
        )

        max_attempts = max(1, self.settings.halo_task_max_retries)
        feedback: str | None = None
        print(f"[halo-apex] target : {rel} (repo: {repo_root})", flush=True)
        print(f"[halo-apex] branch : {branch}", flush=True)

        for attempt in range(1, max_attempts + 1):
            print(
                f"[halo-apex] attempt {attempt}/{max_attempts}: requesting diff ...",
                flush=True,
            )
            try:
                diff_text = orchestrator.request_diff(
                    task, str(file_path), target_dir=str(repo_root), feedback=feedback
                )
            except Exception as exc:
                feedback = f"Attempt {attempt}: LLM request failed: {exc}"
                print(f"[halo-apex] LLM error: {exc}", flush=True)
                continue
            print(
                f"[halo-apex] diff received ({len(diff_text.splitlines())} lines), applying ...",
                flush=True,
            )
            with tempfile.NamedTemporaryFile(
                "w", suffix=".diff", delete=False, encoding="utf-8"
            ) as diff_file:
                diff_file.write(diff_text)
                diff_path = Path(diff_file.name)

            try:
                self.file_ops.apply_patch(str(diff_path), target_dir=str(repo_root))
            except RuntimeError as exc:
                feedback = f"Attempt {attempt}: git apply rejected the diff.\n{exc}"
                print("[halo-apex] git apply rejected the diff, retrying ...", flush=True)
                continue
            finally:
                diff_path.unlink(missing_ok=True)

            # --- Verification gate ----------------------------------------
            gate_error = self._verification_gate(repo_root, file_path)
            if gate_error:
                _git(repo_root, "checkout", "--", str(rel))
                feedback = f"Attempt {attempt}: verification failed.\n{gate_error}"
                print("[halo-apex] verification failed, rolled back, retrying ...", flush=True)
                continue
            print("[halo-apex] verification passed, committing ...", flush=True)

            # --- Success: commit on the task branch ------------------------
            _git(repo_root, "add", str(rel))
            _git(repo_root, "commit", "-m", f"halo-apex: {task}")
            commit = _git(repo_root, "rev-parse", "HEAD").stdout.strip()

            self.index_path(file_path)  # re-index patched content

            return {
                "ok": True,
                "branch": branch,
                "commit": commit,
                "file": str(rel),
                "attempts": attempt,
                "duration_seconds": round(time.time() - started, 1),
                "diffstat": _git(
                    repo_root, "show", "--stat", "--oneline", "HEAD"
                ).stdout.strip(),
                "note": f"Review: git diff {original_branch}...{branch}",
            }

        # --- Retries exhausted: roll back and clean up --------------------
        _git(repo_root, "checkout", "--", str(rel))
        _git(repo_root, "checkout", original_branch)
        _git(repo_root, "branch", "-D", branch)
        raise RuntimeError(
            f"Task failed after {max_attempts} attempts. Rolled back and removed "
            f"branch {branch}.\nLast feedback:\n{feedback}"
        )

    # ------------------------------------------------------------------
    def _verification_gate(self, repo_root: Path, file_path: Path) -> str | None:
        """Return None if the patch passes the gate, else an error text."""
        compile_result = self.executor.run_command(
            f'python3 -m py_compile "{file_path}"', cwd=str(repo_root)
        )
        if compile_result["returncode"] != 0:
            return f"py_compile failed:\n{compile_result['stderr']}"
        if (repo_root / "tests").is_dir():
            test_result = self.executor.run_command(
                "python3 -m pytest -x -q", cwd=str(repo_root)
            )
            if test_result["returncode"] != 0:
                return (
                    f"pytest failed:\n{test_result['stdout']}\n{test_result['stderr']}"
                )
        return None

    # ------------------------------------------------------------------
    def _unique_branch(self, repo_root: Path, slug: str) -> str:
        """Return a branch name that does not exist yet."""
        base = f"{BRANCH_PREFIX}{slug}"
        candidate = base
        counter = 2
        while (
            _git(
                repo_root, "rev-parse", "--verify", candidate, check=False
            ).returncode
            == 0
        ):
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate