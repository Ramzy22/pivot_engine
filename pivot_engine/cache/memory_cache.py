
"""
Simple in-memory cache with TTL support for PyArrow Tables.
"""
import time
from typing import Optional, Any, Dict
import pyarrow as pa

class MemoryCache:
    """
    An in-memory cache for PyArrow Tables with a time-to-live (TTL).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to ensure one cache instance per process
        if not cls._instance:
            cls._instance = super(MemoryCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, ttl: int = 300):
        """
        Initialize the cache.
        
        Args:
            ttl: Default time-to-live for cache entries in seconds.
        """
        self._cache: Dict[str, tuple[pa.Table, float]] = {}
        self.default_ttl = ttl

    def get(self, key: str) -> Optional[pa.Table]:
        """
        Retrieve an Arrow Table from the cache.
        
        Args:
            key: The key of the item to retrieve.
            
        Returns:
            The cached Arrow Table, or None if the item is not found or expired.
        """
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        if time.time() > expiry:
            # Entry has expired
            del self._cache[key]
            return None
            
        return value

    def set(self, key: str, value: pa.Table, ttl: Optional[int] = None):
        """
        Add an Arrow Table to the cache.
        
        Args:
            key: The key of the item to add.
            value: The Arrow Table to add to the cache.
            ttl: Time-to-live for this specific entry. If None, use default.
        """
        if not isinstance(value, pa.Table):
            raise TypeError("MemoryCache can only store PyArrow Tables.")

        ttl_to_use = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl_to_use
        self._cache[key] = (value, expiry)

    def clear(self):
        """Clear all items from the cache."""
        self._cache.clear()

