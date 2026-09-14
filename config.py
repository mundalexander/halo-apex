"""Central Pydantic configuration for the Halo Apex ecosystem.

Loads all HALO_* variables from the environment / ``.env`` file.
The subbrand packages (halo_vector, halo_blade, halo_stream) stay 100%
standalone: they never import this module — the glue code (main.py)
injects these values into them via constructor parameters.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class HaloSettings(BaseSettings):
    """Ecosystem-level configuration; field names map 1:1 to
    upper-case environment variables (see ``.env.example``)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Execution guard --------------------------------------------
    halo_execution_mode: str = "LOCAL_ONLY"

    # --- Halo Vector (embeddings + vector storage) -------------------
    halo_vector_base_url: str = "http://localhost:11434/v1"
    halo_vector_embedding_model: str = "bge-m3"
    halo_vector_persist_dir: str = "./halo_chroma_db"

    # --- Halo Blade (orchestration, LLM endpoint) --------------------
    halo_blade_llm_url: str = "http://localhost:1234/v1"
    halo_blade_llm_model: str = ""          # empty -> auto-detect

    # --- Halo Stream (optional paging interface stub) ----------------
    use_halo_stream: bool = False
    halo_stream_backend_url: str = "http://localhost:1234/v1"

    # --- Misc --------------------------------------------------------
    halo_max_log_lines: int = 50


halo_settings = HaloSettings()