"""
ProgressiveDataLoader - Load data in chunks for progressive rendering
"""
import asyncio
from typing import Dict, Any, Optional, Callable, List
import pyarrow as pa
import ibis
from ibis import BaseBackend as IbisBaseBackend
from ibis.expr.api import Table as IbisTable, Expr
from pivot_engine.types.pivot_spec import PivotSpec


class ProgressiveDataLoader:
    def __init__(self, backend: IbisBaseBackend, cache, event_bus=None):
        self.backend = backend
        self.cache = cache
        self.event_bus = event_bus
        self.default_chunk_size = 1000
        self.min_chunk_size = 100
        
    async def load_progressive_chunks(self, spec: PivotSpec, chunk_callback: Optional[Callable] = None):
        """Load data in chunks for progressive rendering"""
        # Determine chunk boundaries based on data size and complexity
        total_estimated_rows = await self._estimate_total_rows(spec) # Await the async method
        chunk_size = min(self.default_chunk_size, max(self.min_chunk_size, total_estimated_rows // 10))  # Adaptive chunk size
        
        offset = 0
        chunk_number = 0
        
        while True:
            # Fetch chunk
            chunk_ibis_expr = self._build_chunk_ibis_expression(spec, offset, chunk_size)
            chunk_data = await chunk_ibis_expr.to_pyarrow() # Execute Ibis expression
            
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
        root_ibis_expr = await self._create_level_ibis_expression(spec, [], spec.rows[0] if spec.rows else '')
        if root_ibis_expr:
            root_data = await root_ibis_expr.to_pyarrow() # Execute Ibis expression
            
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
                level_ibis_expr = await self._create_level_ibis_expression(spec, path, spec.rows[level])
                if level_ibis_expr:
                    level_data = await level_ibis_expr.to_pyarrow() # Execute Ibis expression
                    
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
    
    async def _estimate_total_rows(self, spec: PivotSpec) -> int:
        """Estimate total number of rows for the query using Ibis."""
        ibis_table = self.backend.table(spec.table)

        # Apply filters
        filtered_table = ibis_table
        if spec.filters:
            filter_expr = self._build_ibis_filter_expression(ibis_table, spec.filters)
            if filter_expr is not None:
                filtered_table = filtered_table.filter(filter_expr)
        
        # Execute count query
        row_count = await filtered_table.count().execute()
        return row_count
    
    def _build_chunk_ibis_expression(self, spec: PivotSpec, offset: int, chunk_size: int) -> IbisTable:
        """Build Ibis expression for a specific chunk."""
        base_table = self.backend.table(spec.table)

        # Apply filters
        filtered_table = base_table
        if spec.filters:
            filter_expr = self._build_ibis_filter_expression(base_table, spec.filters)
            if filter_expr is not None:
                filtered_table = filtered_table.filter(filter_expr)

        # Define aggregations in Ibis
        aggregations = []
        for m in spec.measures:
            agg_func = getattr(filtered_table[m.field], m.agg)
            aggregations.append(agg_func().name(m.alias))

        # Apply grouping
        grouped_table = filtered_table
        if spec.rows:
            grouped_table = filtered_table.group_by(spec.rows)
            
        # Apply aggregation
        agg_expr = grouped_table.aggregate(aggregations)

        # Apply ordering for stable pagination (use grouping columns by default)
        order_cols = spec.rows or [agg_expr.columns[0]] # Order by first grouping col or first agg col
        agg_expr = agg_expr.order_by([ibis.asc(col) for col in order_cols])

        # Apply LIMIT and OFFSET
        agg_expr = agg_expr.limit(chunk_size, offset=offset)
        
        return agg_expr
    
    async def _create_level_ibis_expression(self, base_spec: PivotSpec, parent_path: List[str], current_dimension: str) -> Optional[IbisTable]:
        """Create an Ibis expression for a specific level of the hierarchy"""
        if not current_dimension:
            return None
            
        base_table = self.backend.table(base_spec.table)

        # Build filters based on parent path
        all_filters_dicts = base_spec.filters or []
        for i, value in enumerate(parent_path):
            if i < len(base_spec.rows):
                all_filters_dicts.append({
                    "field": base_spec.rows[i],
                    "op": "=",
                    "value": value
                })
        
        filtered_table = base_table
        if all_filters_dicts:
            filter_expr = self._build_ibis_filter_expression(base_table, all_filters_dicts)
            if filter_expr is not None:
                filtered_table = filtered_table.filter(filter_expr)

        # Define aggregations in Ibis
        aggregations = []
        for measure in base_spec.measures:
            agg_func = getattr(filtered_table[measure.field], measure.agg)
            aggregations.append(agg_func().name(measure.alias))

        # Build the grouped and aggregated expression
        agg_expr = filtered_table.group_by(current_dimension).aggregate(aggregations)

        # Apply ordering
        agg_expr = agg_expr.order_by(ibis.asc(current_dimension))

        # Limit per level (optional, depends on use case)
        agg_expr = agg_expr.limit(1000)
        
        return agg_expr
    
    def _build_ibis_filter_expression(self, table: IbisTable, filters: List[Dict[str, Any]]) -> Optional[Expr]:
        """Builds an Ibis filter expression from a list of filter dictionaries."""
        ibis_filters = []
        for f in filters:
            field = f.get("field")
            op = f.get("op", "=")
            value = f.get("value")

            if field not in table.columns:
                print(f"Warning: Filter field '{field}' not found in table during Ibis filter conversion.")
                continue
            
            col = table[field]
            
            if op in ["=", "=="]:
                ibis_filters.append(col == value)
            elif op == "!=":
                ibis_filters.append(col != value)
            elif op == "<":
                ibis_filters.append(col < value)
            elif op == "<=":
                ibis_filters.append(col <= value)
            elif op == ">":
                ibis_filters.append(col > value)
            elif op == ">=":
                ibis_filters.append(col >= value)
            elif op == "in":
                if isinstance(value, list):
                    ibis_filters.append(col.isin(value))
                else:
                    print(f"Warning: Filter op '{op}' requires a list/tuple/set value for field '{field}'.")
            elif op == "between":
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    ibis_filters.append(col.between(value[0], value[1]))
                else:
                    print(f"Warning: Filter op '{op}' requires a two-element list/tuple value for field '{field}'.")
            elif op == "like":
                ibis_filters.append(col.like(value))
            elif op == "ilike":
                ibis_filters.append(col.ilike(value))
            elif op == "is null":
                ibis_filters.append(col.isnull())
            elif op == "is not null":
                ibis_filters.append(col.notnull())
            elif op == "starts_with":
                ibis_filters.append(col.like(f"{value}%"))
            elif op == "ends_with":
                ibis_filters.append(col.like(f"%{value}"))
            elif op == "contains":
                ibis_filters.append(col.like(f"%{value}%"))
            else:
                print(f"Warning: Unsupported filter operator '{op}' for field '{field}'.")

        if not ibis_filters:
            return None
        
        # Combine all filters with AND
        combined_filter = ibis_filters[0]
        for f_expr in ibis_filters[1:]:
            combined_filter &= f_expr
        return combined_filter
