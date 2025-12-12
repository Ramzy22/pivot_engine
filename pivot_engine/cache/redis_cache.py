"""
Redis-based cache for pivot query results, using Arrow IPC format.
"""
import redis
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import Optional, Any, Dict

class RedisCache:
    """
    A cache implementation that uses Redis as the backend.
    It stores PyArrow Tables using Arrow's IPC format.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl: int = 300):
        """
        Initialize the Redis cache.
        
        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
            ttl: Default time-to-live for cache entries in seconds.
        """
        try:
            # Note: decode_responses=False is important as we are storing binary data
            self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=False)
            self.client.ping()
        except redis.exceptions.ConnectionError as e:
            raise ConnectionError(f"Could not connect to Redis at {host}:{port}. Please ensure Redis is running.") from e
            
        self.default_ttl = ttl

    def get(self, key: str) -> Optional[pa.Table]:
        """
        Retrieve an Arrow Table from the cache.
        
        Args:
            key: The key of the item to retrieve.
            
        Returns:
            The cached Arrow Table, or None if the item is not found.
        """
        cached_value = self.client.get(key)
        if cached_value:
            try:
                # Create a buffer from the bytes and read the table
                buffer = pa.py_buffer(cached_value)
                return ipc.read_table(buffer)
            except pa.lib.ArrowInvalid:
                # Handle cases where the data in cache is not a valid Arrow format
                return None
        return None

    def set(self, key: str, value: pa.Table, ttl: Optional[int] = None):
        """
        Add an Arrow Table to the cache.
        
        Args:
            key: The key of the item to add.
            value: The Arrow Table to add to the cache.
            ttl: Time-to-live for the cache entry in seconds.
                 If not provided, the default TTL is used.
        """
        if not isinstance(value, pa.Table):
            raise TypeError("RedisCache can only store PyArrow Tables.")

        ttl_to_use = ttl if ttl is not None else self.default_ttl
        
        # Serialize the table to a buffer
        buffer = ipc.serialize_table(value).to_pybytes()
        
        self.client.set(key, buffer, ex=ttl_to_use)

    def clear(self):
        """Clear the entire cache."""
        self.client.flushdb()
