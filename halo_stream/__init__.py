"""Halo Stream — optional SSD/RAM/VRAM paging interface (stub).

Halo Stream is disabled by default. When ``USE_HALO_STREAM=False`` every
request passes straight through to the standard LLM URL; Halo Blade and
Halo Vector work 100% independently of this package.
"""

from .provider import HaloStreamProvider

__all__ = ["HaloStreamProvider"]