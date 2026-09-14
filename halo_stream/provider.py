"""Halo Stream — provider stub for the optional paging interface.

Future iterations will use Halo Stream for SSD/RAM/VRAM paging of large
models. Today it is a transparent routing stub:

- Disabled (``USE_HALO_STREAM=False``): every request is routed directly
  to the standard LLM URL (pass-through).
- Enabled: requests are routed to the streaming backend URL instead.
"""

from __future__ import annotations

from typing import Any


class HaloStreamProvider:
    """Unified provider stub with ``stream_status()`` and
    ``get_backend_url()``."""

    def __init__(
        self,
        enabled: bool = False,
        backend_url: str = "",
        default_url: str = "",
    ) -> None:
        self.enabled = enabled
        self.backend_url = backend_url
        self.default_url = default_url

    # ------------------------------------------------------------------
    def stream_status(self) -> dict[str, Any]:
        """Report the current state of the Halo Stream subsystem."""
        return {
            "available": self.enabled,
            "routing": "halo-stream" if self.enabled else "direct",
            "mode": "ssd/ram/vram paging (stub)",
            "backend_url": self.backend_url if self.enabled else None,
        }

    # ------------------------------------------------------------------
    def get_backend_url(self, default_url: str | None = None) -> str:
        """Return the effective LLM backend URL.

        With Halo Stream disabled this transparently returns the
        standard LLM URL (pass-through routing). With Halo Stream enabled
        it returns the paging/streaming backend URL instead.
        """
        fallback = default_url if default_url is not None else self.default_url
        if not self.enabled:
            return fallback
        return self.backend_url or fallback