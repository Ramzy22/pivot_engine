"""
ProgressiveDataLoader - Load data in chunks for progressive rendering
"""
import asyncio
from typing import Dict, Any, Optional, Callable, List
import pyarrow as pa
from pivot_engine.types.pivot_spec import PivotSpec


class ProgressiveDataLoader:
    def __init__(self, backend, cache, event_bus=None):
        self.backend = backend
        self.cache = cache
        self.event_bus = event_bus
        self.default_chunk_size = 1000
        self.min_chunk_size = 100
        
    async def load_progressive_chunks(self, spec: PivotSpec, chunk_callback: Optional[Callable] = None):
        """Load data in chunks for progressive rendering"""
        # Determine chunk boundaries based on data size and complexity
        total_estimated_rows = self._estimate_total_rows(spec)
        chunk_size = min(self.default_chunk_size, max(self.min_chunk_size, total_estimated_rows // 10))  # Adaptive chunk size
        
        offset = 0
        chunk_number = 0
        
        while True:
            # Fetch chunk
            chunk_query = self._build_chunk_query(spec, offset, chunk_size)
            chunk_data = await self.backend.execute(chunk_query)
            
            if chunk_data.num_rows == 0:
                break
                
            # Notify UI about chunk availability
            chunk_info = {
                'data': chunk_data,
                'offset': offset,
                'total_estimated': total_estimated_rows,
                'progress': min(1.0, (offset + chunk_size) / total_estimated_rows),
                'chunk_number': chunk_number,
                'is_last_chunk': chunk_data.num_rows < chunk_size
            }
            
            if chunk_callback:
                await chunk_callback(chunk_info)
            
            offset += chunk_size
            chunk_number += 1
            
            if chunk_data.num_rows < chunk_size:
                # Last chunk
                break
        
        return {'total_chunks': chunk_number, 'total_rows': offset}
    
    async def load_hierarchical_progressive(self, spec: PivotSpec, expanded_paths: List[List[str]], level_callback: Optional[Callable] = None):
        """Load hierarchical data progressively by levels"""
        result = {'levels': []}
        
        # Load root level first (top-level aggregations)
        root_spec = await self._create_level_spec(spec, [], spec.rows[0] if spec.rows else '')
        if root_spec:
            root_data = await self.backend.execute(root_spec)
            
            level_info = {
                'level': 0,
                'data': root_data,
                'is_root': True,
                'total_rows': root_data.num_rows if root_data else 0
            }
            
            result['levels'].append(level_info)
            
            if level_callback:
                await level_callback(level_info)
        
        # Load expanded paths progressively
        for path in expanded_paths:
            level = len(path)
            if level < len(spec.rows):
                level_spec = await self._create_level_spec(spec, path, spec.rows[level])
                if level_spec:
                    level_data = await self.backend.execute(level_spec)
                    
                    level_info = {
                        'level': level,
                        'data': level_data,
                        'parent_path': path,
                        'is_expanded': True,
                        'total_rows': level_data.num_rows if level_data else 0
                    }
                    
                    result['levels'].append(level_info)
                    
                    if level_callback:
                        await level_callback(level_info)
        
        return result
    
    def _estimate_total_rows(self, spec: PivotSpec) -> int:
        """Estimate total number of rows for the query"""
        # Use a count query to estimate
        count_query = f"SELECT COUNT(*) as row_count FROM {spec.table}"

        if spec.filters:
            # Add WHERE clause based on filters
            where_parts = []
            params = []
            for f in spec.filters:
                field = f.get("field")
                op = f.get("op", "=")
                val = f.get("value")
                where_parts.append(f"{field} {op} ?")
                params.append(val)
            count_query += " WHERE " + " AND ".join(where_parts)

        result = self.backend.execute({"sql": count_query, "params": params})
        if result and result.num_rows > 0:
            return result.column(0)[0].as_py()
        return 0
    
    def _build_chunk_query(self, spec: PivotSpec, offset: int, chunk_size: int) -> Dict[str, Any]:
        """Build query for a specific chunk"""
        # Build the main query with pagination
        select_parts = []
        select_parts.extend(spec.rows)
        
        for measure in spec.measures:
            select_parts.append(f"{measure.agg}({measure.field}) as {measure.alias}")
        
        group_by_clause = ", ".join(spec.rows) if spec.rows else ""
        where_clause = self._build_where_clause(spec.filters)
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM {spec.table}
        {where_clause}
        """
        
        if group_by_clause:
            query += f" GROUP BY {group_by_clause}"
        
        # Add LIMIT and OFFSET
        query += f" LIMIT {chunk_size} OFFSET {offset}"
        
        return {"sql": query, "params": self._extract_params(spec.filters)}
    
    async def _create_level_spec(self, base_spec: PivotSpec, parent_path: List[str], current_dimension: str) -> Optional[Dict[str, Any]]:
        """Create a pivot spec for a specific level of the hierarchy"""
        if not current_dimension:
            return None
            
        # Build filters based on parent path
        all_filters = base_spec.filters or []
        for i, value in enumerate(parent_path):
            if i < len(base_spec.rows):
                all_filters.append({
                    "field": base_spec.rows[i],
                    "op": "=",
                    "value": value
                })
        
        # Build the query for this specific level
        select_parts = [current_dimension]
        for measure in base_spec.measures:
            select_parts.append(f"{measure.agg}({measure.field}) as {measure.alias}")
        
        where_clause = self._build_where_clause(all_filters)
        params = self._extract_params(all_filters)
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM {base_spec.table}
        {where_clause}
        GROUP BY {current_dimension}
        ORDER BY {current_dimension}
        LIMIT 1000  -- Limit per level
        """
        
        return {"sql": query, "params": params}
    
    def _build_where_clause(self, filters: List[Dict[str, Any]]) -> str:
        """Build WHERE clause from filters"""
        if not filters:
            return ""
        
        conditions = []
        for f in filters:
            field = f.get("field", "")
            op = f.get("op", "=")
            # Note: The actual value is handled through parameters
            conditions.append(f"{field} {op} ?")
        
        return "WHERE " + " AND ".join(conditions)
    
    def _extract_params(self, filters: List[Dict[str, Any]]) -> List[Any]:
        """Extract parameters from filters"""
        params = []
        for f in filters:
            params.append(f.get("value"))
        return params