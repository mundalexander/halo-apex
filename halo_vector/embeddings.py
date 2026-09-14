"""Halo Vector — embedding service for local bge-m3 embeddings.

Talks to any OpenAI-compatible embedding endpoint (Ollama, LM Studio,
llama.cpp, ...) via the ``EMBEDDING_BASE_URL`` it is given at
construction time. Zero dependencies on other Halo subbrands.
"""

from __future__ import annotations

from openai import OpenAI


class EmbeddingService:
    """OpenAI-compatible embedding client (default target: bge-m3)."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "not-needed",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = OpenAI(
            base_url=self.base_url, api_key=api_key, timeout=timeout
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text and return its embedding vector."""
        response = self.client.embeddings.create(
            input=[text], model=self.model_name
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per text."""
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=list(texts), model=self.model_name
        )
        return [item.embedding for item in response.data]

    def health_check(self) -> bool:
        """True if a real embedding round-trip with the configured model
        succeeds."""
        try:
            self.embed_text("halo vector health check")
            return True
        except Exception:
            return False