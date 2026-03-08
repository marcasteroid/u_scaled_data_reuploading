"""
Compatibility wrapper for cache utilities.

Canonical implementation currently lives in residuals/utils/cache.py.
"""

from residuals.utils.cache import _hash_array, _kernel_block_cached, _safe_psd_hygiene, memory

__all__ = ["memory", "_hash_array", "_safe_psd_hygiene", "_kernel_block_cached"]

