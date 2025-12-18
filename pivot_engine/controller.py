"""
PivotController - Enhanced with support for advanced pivot features.
"""
from typing import Optional, Any, Dict, List, Union, Callable, Generator
import time
import decimal
import pyarrow as pa
import ibis
from ibis.expr.api import Table as IbisTable

from .tree import TreeExpansionManager
from .planner.sql_planner import SQLPlanner
from .planner.ibis_planner import IbisPlanner
from .diff.diff_engine import QueryDiffEngine
from .backends.duckdb_backend import DuckDBBackend
from .backends.ibis_backend import IbisBackend
from .cache.memory_cache import MemoryCache
from .cache.redis_cache import RedisCache
from .types.pivot_spec import PivotSpec

def sanitize_column_name(value: str) -> str:
    """Sanitize column value for use in SQL identifier"""
    import re
    if not value:
        return "null"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(value))
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized[:63]

class PivotController:
    """
    Enhanced controller for pivot operations with advanced features.
    Coordinates: Enhanced Planner -> Enhanced DiffEngine -> Cache -> Backend
    """
    def __init__(
        self,
        backend_uri: str = ":memory:",
        cache: Union[str, Any] = "memory",
        planner: Optional[Any] = None,
        planner_name: str = "ibis",
        enable_tiles: bool = True,
        enable_delta: bool = True,
        tile_size: int = 100,
        cache_ttl: int = 300,
        **cache_options: Any
    ):
        self.enable_arrow = True
        if backend_uri == ":memory:":
            backend_uri = ":memory:shared_pivot_db"

        # Initialize Cache
        if isinstance(cache, str):
            if cache == "redis":
                self.cache = RedisCache(**cache_options)
            elif cache == "memory":
                self.cache = MemoryCache(ttl=cache_ttl)
            else:
                raise ValueError(f"Unknown cache type: {cache}")
        else:
            self.cache = cache or MemoryCache(ttl=cache_ttl)

        # Initialize Backend and Planner
        self.backend = None
        self.planner = planner

        if not self.planner:
            if planner_name == "ibis":
                try:
                    # Use IbisBackend which handles connection details
                    self.backend = IbisBackend(connection_uri=backend_uri)
                    # Create IbisPlanner with the connection from the backend
                    self.planner = IbisPlanner(con=self.backend.con)
                except Exception as e:
                    print(f"Could not connect to database backend via Ibis: {e}, falling back to SQLPlanner/DuckDB")
                    self.backend = DuckDBBackend(uri=backend_uri)
                    self.planner = SQLPlanner(dialect="duckdb")
            else:
                self.backend = DuckDBBackend(uri=backend_uri)
                self.planner = SQLPlanner(dialect="duckdb")
        else:
            # If planner is provided, try to infer backend or create a default one
            # This path assumes the caller manages the backend/planner relationship
            if not self.backend:
                self.backend = DuckDBBackend(uri=backend_uri)

        # The diff_engine expects an Ibis connection for _compute_true_total
        # If we are using IbisBackend, we pass its connection
        # If using DuckDBBackend, we might need to handle it differently or DiffEngine supports it
        diff_engine_backend = self.backend.con if isinstance(self.backend, IbisBackend) else None
        
        self.diff_engine = QueryDiffEngine(
            cache=self.cache,
            default_ttl=cache_ttl,
            tile_size=tile_size,
            enable_tiles=enable_tiles,
            enable_delta_updates=enable_delta,
            backend=diff_engine_backend
        )
        self.tree_manager = TreeExpansionManager(self)

        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def run_hierarchical_pivot(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self.tree_manager.run_hierarchical_pivot(spec)

    def toggle_expansion(self, spec_hash: str, path: List[str]) -> Dict[str, Any]:
        return self.tree_manager.toggle_expansion(spec_hash, path)

    def run_pivot(
        self,
        spec: Any,
        return_format: str = "arrow",
        force_refresh: bool = False
    ) -> Union[Dict[str, Any], pa.Table]:
        self._request_count += 1
        spec = self._normalize_spec(spec)
        plan_result = self.planner.plan(spec) # Returns {"queries": List[IbisExpr], "metadata": {...}}
        metadata = plan_result.get("metadata", {})
        queries_to_run = plan_result.get("queries", [])

        # Queries in plan_result.queries are now Ibis expressions
        # The logic here needs to adapt to handle Ibis expressions directly

        if metadata.get("needs_column_discovery"):
            result_table = self._execute_topn_pivot(spec, plan_result, force_refresh)
        else:
            result_table = self._execute_standard_pivot(spec, plan_result, force_refresh)

        # Final conversion based on requested format
        if return_format == "dict":
            return self._convert_table_to_dict(result_table, spec)
            
        return result_table

    def _execute_standard_pivot(self, spec: Any, plan_result: Dict[str, Any], force_refresh: bool) -> pa.Table:
        # The diff_engine.plan now returns Ibis expressions
        queries_to_run, strategy = self.diff_engine.plan(plan_result, spec, force_refresh=force_refresh)

        self._cache_hits += strategy.get("cache_hits", 0)
        self._cache_misses += len(queries_to_run)

        results = []
        for query_ibis_expr in queries_to_run:
            # Execute the Ibis expression directly
            results.append(query_ibis_expr.to_pyarrow())
            
        # Get the main aggregation result if available
        main_result = results[0] if results else pa.table({}) if pa is not None else None

        # Compute totals using Arrow operations if needed
        metadata = plan_result.get("metadata", {})
        if metadata.get("needs_totals", False) and main_result is not None and main_result.num_rows > 0:
            # Calculate totals from the main result using Arrow compute
            main_result = self._compute_totals_arrow(main_result, metadata)

        # The diff_engine.merge_and_finalize also needs to be updated for Ibis expressions
        # For now, it will receive pyarrow tables
        final_table = self.diff_engine.merge_and_finalize([main_result] if main_result is not None else [], plan_result, spec, strategy)

        return main_result if main_result is not None else pa.table({}) if pa is not None else None

    def _execute_topn_pivot(self, spec: Any, plan_result: Dict[str, Any], force_refresh: bool) -> pa.Table:
        """Execute top-N pivot"""
        # queries now contains Ibis expressions
        queries = plan_result.get("queries", [])
        col_ibis_expr = queries[0] # This should be an Ibis expression
        
        # The cache key needs to adapt to Ibis expressions
        col_cache_key = self._cache_key_for_query(col_ibis_expr, spec)
        cached_cols_table = self.cache.get(col_cache_key) if not force_refresh else None

        if cached_cols_table:
            column_values = cached_cols_table.column("_col_key").to_pylist()
            self._cache_hits += 1
        else:
            # Execute the Ibis expression directly
            col_results_table = col_ibis_expr.to_pyarrow()
            column_values = col_results_table.column("_col_key").to_pylist()
            self.cache.set(col_cache_key, col_results_table)
            self._cache_misses += 1

        # planner.build_pivot_query_from_columns now returns an Ibis expression
        pivot_ibis_expr = self.planner.build_pivot_query_from_columns(spec, column_values)
        
        pivot_cache_key = self._cache_key_for_query(pivot_ibis_expr, spec)
        cached_pivot_table = self.cache.get(pivot_cache_key) if not force_refresh else None

        if cached_pivot_table:
            result_table = cached_pivot_table
        else:
            # Execute the Ibis expression directly
            pivot_results_table = pivot_ibis_expr.to_pyarrow()
            self.cache.set(pivot_cache_key, pivot_results_table)
            result_table = pivot_results_table

        # Compute totals using Arrow operations if needed
        metadata = plan_result.get("metadata", {})
        if metadata.get("needs_totals", False) and result_table is not None and result_table.num_rows > 0:
            result_table = self._compute_totals_arrow(result_table, metadata)

        return result_table

    def load_data_from_arrow(
        self,
        table_name: str,
        arrow_table: pa.Table,
        register_checkpoint: bool = True
    ):
        # Use the Ibis connection to create tables from Arrow, making them visible to the IbisPlanner
        self.planner.con.create_table(table_name, arrow_table, overwrite=True)
        if register_checkpoint:
            self.register_delta_checkpoint(table_name, timestamp=time.time())

    def register_delta_checkpoint(self, table: str, timestamp: float = None, max_id: Optional[int] = None, incremental_field: str = "updated_at"):
        timestamp = timestamp or time.time()
        # The diff_engine.register_delta_checkpoint needs to be updated to work with Ibis as well
        self.diff_engine.register_delta_checkpoint(table, timestamp, max_id, incremental_field)

    def _normalize_spec(self, spec: Any) -> Any:
        if isinstance(spec, dict):
            return PivotSpec.from_dict(spec)
        return spec

    def _convert_table_to_dict(self, table: Optional[pa.Table], spec: PivotSpec) -> Dict[str, Any]:
        """Convert a PyArrow Table to the legacy dictionary format."""
        if table is None or table.num_rows == 0:
            return {"columns": [], "rows": [], "next_cursor": None}

        data_as_dicts = table.to_pylist()
        # Convert Decimal and other non-JSON-serializable types to basic types
        rows_as_lists = []
        for row in data_as_dicts:
            row_list = []
            for value in row.values():
                if hasattr(value, 'as_py'):  # Arrow scalar
                    value = value.as_py()
                # Handle non-JSON serializable types
                if isinstance(value, (decimal.Decimal,)):
                    value = float(value)
                elif value is None or (isinstance(value, float) and (value != value)):  # NaN check
                    value = None
                row_list.append(value)
            rows_as_lists.append(row_list)

        next_cursor = None
        if spec.limit and table.num_rows == spec.limit: # Check if result size matches limit
            next_cursor = self._generate_next_cursor(table, spec)

        return {
            "columns": table.column_names,
            "rows": rows_as_lists,
            "next_cursor": next_cursor
        }

    def _compute_totals_arrow(self, table: pa.Table, metadata: Dict[str, Any]) -> pa.Table:
        """
        Compute totals using Arrow compute operations for efficiency.
        This method computes grand totals from the main result table.
        """
        if table.num_rows == 0:
            return table

        import pyarrow.compute as pc
        import pyarrow as pa

        # Get the aggregation aliases and their original aggregation types from measures
        agg_aliases = metadata.get("agg_aliases", [])

        # For more accurate totals computation, we need to know the original aggregation types
        # However, since we're computing totals from aggregated data, we'll assume SUM for most cases
        # unless we can determine the original measure types from the metadata

        # Calculate totals for each aggregation column
        total_values = {}
        for col_name in table.column_names:
            if col_name in agg_aliases:
                col_array = table.column(col_name)

                # For aggregated values, use SUM to calculate grand totals
                # This is appropriate for SUM, COUNT, etc. aggregations
                # For AVG, we would need special handling (weighted average), but that's complex
                try:
                    if pa.types.is_integer(col_array.type) or pa.types.is_floating(col_array.type):
                        total_val = pc.sum(col_array).as_py()
                    elif pa.types.is_decimal(col_array.type):
                        total_val = pc.sum(col_array).as_py()
                    else:
                        total_val = pc.sum(col_array).as_py()
                        if total_val is None:
                            total_val = col_array[0].as_py() if len(col_array) > 0 else None
                except Exception:
                    # If sum fails, return the original values or None
                    total_val = col_array[0].as_py() if len(col_array) > 0 else None

                total_values[col_name] = [total_val]
            else:
                # For non-aggregation (grouping) columns, set to None in totals row
                total_values[col_name] = [None]

        # Create a new row for the totals with proper schema
        total_row_arrays = []
        for col_name in table.column_names:
            if col_name in total_values:
                # Use the same data type as the original column
                value = total_values[col_name][0]
                if value is None:
                    # Create null array of the appropriate type
                    total_row_arrays.append(pa.array([None], type=table.schema.field(col_name).type))
                else:
                    # Create array of the same type as the original
                    original_type = table.schema.field(col_name).type
                    total_row_arrays.append(pa.array([value], type=original_type))
            else:
                # For any missing columns, use null
                total_row_arrays.append(pa.array([None], type=pa.string()))

        # Create totals table with the same schema as original
        try:
            totals_table = pa.table(total_row_arrays, schema=table.schema)
        except Exception:
            # Fallback: ensure the schema matches
            totals_table = pa.table(total_row_arrays, names=table.column_names)

        # Concatenate the original table with the totals row
        return pa.concat_tables([table, totals_table])

    def _generate_next_cursor(self, table: pa.Table, spec: PivotSpec) -> Optional[Dict[str, Any]]:
        """Generate the cursor for the next page based on the last row."""
        if not spec.sort or table.num_rows == 0:
            return None

        sort_keys = spec.sort if isinstance(spec.sort, list) else [spec.sort]
        last_row = table.to_pylist()[-1]
        
        cursor = {}
        for key in sort_keys:
            field = key.get("field")
            if field in last_row:
                cursor[field] = last_row[field]
        
        return cursor if cursor else None

    def _cache_key_for_query(self, ibis_expr: IbisTable, spec: Any) -> str:
        """Generate cache key for an Ibis expression."""
        import json
        import hashlib
        
        # Use Ibis expression hash or compiled string for the key
        # Compiling to SQL here might be expensive just for caching, but ensures uniqueness
        # Alternatively, use a structural hash of the Ibis expression object if Ibis provides one
        try:
            # Attempt to compile the expression to SQL for a unique string representation
            # This is a robust way to get a unique key for the query itself
            compiled_sql = str(self.planner.con.compile(ibis_expr))
            # Include the spec hash as well to ensure filter/sort context is captured
            spec_hash = hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]
            key_str = f"{compiled_sql}-{spec_hash}"
            key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:32]
            return f"pivot_ibis:query:{key_hash}"
        except Exception as e:
            # Fallback if compilation fails (e.g., incomplete expression)
            print(f"Warning: Could not compile Ibis expression for cache key: {e}")
            return f"pivot_ibis:query_fallback:{hashlib.sha256(str(ibis_expr).encode()).hexdigest()[:32]}"

    def run_hierarchical_pivot_progressive(
        self,
        spec: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Build and return tree progressively with intermediate results"""
        return self.tree_manager.run_hierarchical_pivot_progressive(spec, progress_callback)

    def run_hierarchical_pivot_streaming(
        self,
        spec: Dict[str, Any],
        path_cursor_map: Optional[Dict[str, Dict[str, Any]]] = None,
        chunk_size: int = 1000
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream hierarchical pivot results in chunks"""
        # Use DuckDB's fetchmany() for chunked results
        spec_hash = self.tree_manager._hash_spec(spec)
        dimension_hierarchy = spec.get("rows", [])

        # Process and yield chunks instead of building full tree
        for chunk in self.tree_manager._build_tree_chunks(spec, path_cursor_map, chunk_size):
            yield chunk

    def run_hierarchical_pivot_with_prefetch(
        self,
        spec: Dict[str, Any],
        path_cursor_map: Optional[Dict[str, Dict[str, Any]]] = None,
        prefetch_depth: int = 1
    ) -> Dict[str, Any]:
        """Run hierarchical pivot with optional prefetching of expanded nodes"""
        return self.tree_manager.run_hierarchical_pivot_with_prefetch(spec, path_cursor_map, prefetch_depth)

    def clear_cache(self):
        """Clear all cached queries to force fresh data retrieval"""
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()

    def close(self):
        """Close any resources held by the controller"""
        if hasattr(self, 'backend') and hasattr(self.backend, 'close'):
            self.backend.close()

    def run_pivot_arrow(
        self,
        spec: Any,
    ) -> pa.Table:
        """
        Execute a pivot query and return the result as a PyArrow Table.
        This method is used by the Flight server for Arrow-native operations.
        """
        # Execute the pivot query and return the raw Arrow table
        result = self.run_pivot(spec, return_format="arrow")
        if isinstance(result, pa.Table):
            return result
        else:
            # If for some reason it's not an Arrow table, convert it
            # This would typically be the case if some error handling returns different format
            raise ValueError(f"Expected PyArrow Table but got {type(result)}")

    def run_hierarchical_pivot_batch_load(
        self,
        spec: Dict[str, Any],
        expanded_paths: List[List[str]],
        max_levels: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run batch loading of multiple levels of the hierarchy"""
        return self.tree_manager._load_multiple_levels_batch(spec, expanded_paths, max_levels)