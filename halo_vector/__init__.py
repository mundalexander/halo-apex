"""Halo Vector — standalone local embeddings (bge-m3) & vector storage.

Fully self-contained package: it can be lifted into its own repository
and imported as an independent Python package. No imports from other
Halo subbrands or ecosystem glue code.
"""

from .embeddings import EmbeddingService
from .vector_db import VectorEngine

__all__ = ["EmbeddingService", "VectorEngine"]