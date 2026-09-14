"""Halo Blade — OpenClaw engine core: task orchestration.

Halo Blade is 100% standalone: it never imports halo_vector or
halo_stream. Vector retrieval results are injected through the
``VectorSearch`` protocol (structural typing / duck typing), so any
engine that provides ``query()`` plugs in seamlessly — the canonical
implementation being ``halo_vector.VectorEngine``.

The LLM interface is fully agnostic: any OpenAI-compatible endpoint
works behind ``llm_base_url`` — LM Studio, Ollama, llama.cpp, or a
Halo Stream router in front of any of them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI

from .code_ast import ASTCodeParser

SYSTEM_PROMPT = """\
You are an autonomous coding agent inside the Halo Blade engine.
You answer with a single Git unified diff that performs exactly the
requested change. Output rules (strictly):

- Output ONLY the unified diff. No prose, no explanations, no markdown
  fences, no <think> blocks.
- Use the exact target file path given in the task with prefixes "a/"
  and "b/".
- Include 3 lines of surrounding context in every hunk (never produce
  zero-context hunks).
- Keep context lines byte-identical to the numbered input file.
- Keep the diff minimal: change only what the task requires.
"""

# Name markers that identify embedding models; they are never valid
# chat-completion models and must be skipped during auto-detection.
_EMBEDDING_MARKERS = ("embed", "bge")


@runtime_checkable
class VectorSearch(Protocol):
    """Retrieval interface expected from a pluggable vector engine.

    ``halo_vector.VectorEngine`` satisfies this protocol out of the box.
    """

    def query(self, query_text: str, n_results: int = 3) -> list[dict[str, Any]]: ...


class OpenClawOrchestrator:
    """Combines task, AST skeleton and injected vector-retrieval context
    and asks the configured OpenAI-compatible LLM for a minimal Git
    unified diff."""

    def __init__(
        self,
        llm_base_url: str,
        llm_model: str = "",
        vector_engine: VectorSearch | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.llm_base_url = llm_base_url.rstrip("/")
        self.ast_parser = ASTCodeParser()
        self.vector_engine = vector_engine
        self.llm = OpenAI(
            base_url=self.llm_base_url, api_key="not-needed", timeout=timeout
        )
        self._llm_model: str | None = llm_model or None

    # ------------------------------------------------------------------
    def _resolve_llm_model(self) -> str:
        """Return the configured chat model or auto-detect the first
        available non-embedding model on the endpoint."""
        if self._llm_model:
            return self._llm_model
        models = self.llm.models.list()
        candidates = [
            m.id
            for m in models.data
            if not any(marker in m.id.lower() for marker in _EMBEDDING_MARKERS)
        ]
        if not candidates:
            raise RuntimeError(
                "No chat-capable LLM model available on the endpoint."
            )
        self._llm_model = candidates[0]
        return self._llm_model

    # ------------------------------------------------------------------
    def build_context(
        self,
        task: str,
        target_file: str,
        target_dir: str | None = None,
    ) -> str:
        """Assemble task + AST skeleton + vector-retrieval context."""
        skeleton = self.ast_parser.extract_skeleton(target_file)

        content = Path(target_file).read_text(encoding="utf-8")
        numbered = "\n".join(
            f"{i:>4} | {line}"
            for i, line in enumerate(content.splitlines(), start=1)
        )

        if target_dir:
            rel_path = str(Path(target_file).relative_to(target_dir))
        else:
            rel_path = Path(target_file).name

        snippets: list[dict[str, Any]] = (
            self.vector_engine.query(task, n_results=3)
            if self.vector_engine is not None
            else []
        )
        snippet_blocks: list[str] = []
        for hit in snippets:
            meta = hit.get("metadata") or {}
            similarity = hit.get("similarity")
            sim_suffix = (
                f", similarity {similarity:.3f}"
                if isinstance(similarity, float)
                else ""
            )
            snippet_blocks.append(
                f"[{meta.get('file_name', 'snippet')} lines "
                f"{meta.get('line_start', '?')}-{meta.get('line_end', '?')}"
                f"{sim_suffix}]\n{hit.get('text', '')}"
            )
        retrieval = "\n\n".join(snippet_blocks) or "(no indexed context)"

        return (
            f"## Task\n{task}\n\n"
            f"## Target file: {rel_path}\n\n"
            f"## AST skeleton (Halo Blade)\n```\n{skeleton}\n```\n\n"
            f"## Retrieved code context (Halo Vector)\n"
            f"```\n{retrieval}\n```\n\n"
            f"## Full file content (numbered)\n```\n{numbered}\n```\n\n"
            f"Now produce the unified diff for '{rel_path}'."
        )

    # ------------------------------------------------------------------
    def request_diff(
        self,
        task: str,
        target_file: str,
        target_dir: str | None = None,
    ) -> str:
        """Ask the LLM for a unified diff that performs ``task`` on
        ``target_file`` and return the normalized diff text."""
        prompt = self.build_context(task, target_file, target_dir)
        response = self.llm.chat.completions.create(
            model=self._resolve_llm_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return self.normalize_diff(raw)

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_diff(text: str) -> str:
        """Strip reasoning blocks, prose and markdown fences from an LLM
        answer and return the raw unified diff."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()

        match = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        for marker in ("diff --git ", "--- "):
            idx = text.find(marker)
            if idx > 0:
                text = text[idx:]
                break

        return text if text.endswith("\n") else text + "\n"