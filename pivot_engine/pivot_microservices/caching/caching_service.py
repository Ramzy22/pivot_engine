"""
caching_service.py - Distributed caching service for pivot results
"""
import asyncio
import json
import hashlib
from typing import Dict, Any, Optional, Union
import pyarrow as pa
from pivot_engine.cache.memory_cache import MemoryCache
from pivot_engine.cache.redis_cache import RedisCache
from pivot_engine.types.pivot_spec import PivotSpec


class CacheService:
    """Distributed caching service for pivot results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ttl = config.get('default_ttl', 300)
        self.levels = {
            'l1': MemoryCache(ttl=config.get('l1_ttl', 60)),  # Local/In-memory cache
            'l2': self._create_distributed_cache(config)  # Distributed cache
        }
        self.compression_enabled = config.get('compression', True)
        
    def _create_distributed_cache(self, config: Dict[str, Any]):
        """Create distributed cache based on configuration"""
        cache_type = config.get('cache_type', 'memory')
        
        if cache_type == 'redis':
            redis_config = config.get('redis_config', {})
            return RedisCache(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                ttl=self.ttl
            )
        else:
            # Default to memory cache if no distributed cache
            return MemoryCache(ttl=self.ttl)
    
    async def get_or_compute(self, key: str, compute_func, ttl: Optional[int] = None, 
                           force_refresh: bool = False) -> Any:
        """Get value from cache or compute and store it"""
        ttl = ttl or self.ttl
        
        # Try L1 cache first
        if not force_refresh:
            result = await self._get_from_level('l1', key)
            if result is not None:
                return result
        
        # Try L2 cache
        if not force_refresh:
            result = await self._get_from_level('l2', key)
            if result is not None:
                # Populate L1 cache
                await self._store_in_level('l1', key, result, ttl)
                return result
        
        # Compute the value
        result = await compute_func()
        
        # Store in both levels
        await self._store_in_level('l2', key, result, ttl)
        await self._store_in_level('l1', key, result, min(ttl, 60))  # L1 has shorter TTL
        
        return result
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from any cache level"""
        # Try L1 first
        result = await self._get_from_level('l1', key)
        if result is not None:
            return result
        
        # Try L2
        result = await self._get_from_level('l2', key)
        if result is not None:
            # Populate L1 as it was a hit in L2
            await self._store_in_level('l1', key, result, min(self.ttl, 60))
            return result
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all cache levels"""
        ttl = ttl or self.ttl
        success = True
        
        # Store in all levels
        success &= await self._store_in_level('l1', key, value, min(ttl, 60))
        success &= await self._store_in_level('l2', key, value, ttl)
        
        return success
    
    async def _get_from_level(self, level: str, key: str) -> Optional[Any]:
        """Get value from specific cache level"""
        cache = self.levels[level]
        
        if hasattr(cache, 'get'):
            if asyncio.iscoroutinefunction(cache.get):
                return await cache.get(key)
            else:
                return cache.get(key)
        return None
    
    async def _store_in_level(self, level: str, key: str, value: Any, ttl: int) -> bool:
        """Store value in specific cache level"""
        cache = self.levels[level]
        
        # Compress large values before storing
        if self.compression_enabled and self._should_compress(value):
            import pickle
            import zlib
            serialized_value = pickle.dumps(value)
            compressed_value = zlib.compress(serialized_value)
            value_to_store = compressed_value
        else:
            value_to_store = value
        
        if hasattr(cache, 'set'):
            if asyncio.iscoroutinefunction(cache.set):
                await cache.set(key, value_to_store, ttl=ttl)
            else:
                cache.set(key, value_to_store, ttl=ttl)
            return True
        return False
    
    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed"""
        # Simple heuristic: compress if it's large
        if isinstance(value, pa.Table):
            # PyArrow tables can be large
            return value.num_rows > 1000 or value.nbytes > 1024 * 100  # 100KB
        elif isinstance(value, (list, dict)):
            import sys
            return sys.getsizeof(str(value)) > 1024 * 50  # 50KB
        return False
    
    def _decompress_if_needed(self, value: Any):
        """Decompress value if it was compressed"""
        if isinstance(value, bytes):
            try:
                import pickle
                import zlib
                decompressed = zlib.decompress(value)
                return pickle.loads(decompressed)
            except:
                return value  # Return original if decompression fails
        return value
    
    async def invalidate(self, key_pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        invalidated_count = 0
        
        # For simplicity, we'll clear all matching keys
        # In a real implementation, we'd use Redis pattern matching
        if key_pattern == '*':  # Clear all
            if hasattr(self.levels['l1'], 'clear'):
                if asyncio.iscoroutinefunction(self.levels['l1'].clear):
                    await self.levels['l1'].clear()
                else:
                    self.levels['l1'].clear()
            if hasattr(self.levels['l2'], 'clear'):
                if asyncio.iscoroutinefunction(self.levels['l2'].clear):
                    await self.levels['l2'].clear()
                else:
                    self.levels['l2'].clear()
            invalidated_count = -1  # All entries invalidated
        else:
            # Specific key invalidation
            await self._remove_from_level('l1', key_pattern)
            await self._remove_from_level('l2', key_pattern)
            invalidated_count = 1
        
        return invalidated_count
    
    async def _remove_from_level(self, level: str, key: str):
        """Remove specific key from cache level"""
        # This assumes a method exists or we'll use a workaround
        # For now, we'll just override with None and short TTL
        await self._store_in_level(level, key, None, 1)
    
    def generate_cache_key(self, spec: PivotSpec, additional_parts: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for a pivot specification"""
        import json
        
        spec_dict = spec.to_dict()
        
        # Create a hash of the spec to ensure uniqueness
        key_parts = {
            'table': spec.table,
            'rows': sorted(spec.rows),
            'columns': sorted(spec.columns),
            'measures': sorted([m.alias for m in spec.measures]),
            'filters_hash': hashlib.sha256(json.dumps(spec.filters, sort_keys=True).encode()).hexdigest()[:16],
            'additional': additional_parts or {}
        }
        
        key_json = json.dumps(key_parts, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]
        
        return f"pivot:{key_hash}"
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        
        for level_name, cache in self.levels.items():
            if hasattr(cache, 'get_stats'):
                if asyncio.iscoroutinefunction(cache.get_stats):
                    stats[level_name] = await cache.get_stats()
                else:
                    stats[level_name] = cache.get_stats()
            else:
                stats[level_name] = {'size': 'unknown'}
        
        return stats


class DistributedCacheService:
    """Higher-level service for distributed caching with additional features"""
    
    def __init__(self, caching_service: CacheService):
        self.caching_service = caching_service
        self.pipelines = {}  # Cache pipelines for different use cases
        self.subscribers = []  # For cache change notifications