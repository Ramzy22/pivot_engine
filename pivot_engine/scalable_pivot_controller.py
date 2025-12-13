"""
ScalablePivotController - Main controller for high-scale pivot operations
"""
from typing import Optional, Any, Dict, List, Union, Callable, Generator
import time
import decimal
import pyarrow as pa
import asyncio
from .tree import TreeExpansionManager
from .planner.sql_planner import SQLPlanner
from .planner.ibis_planner import IbisPlanner
from .diff.diff_engine import QueryDiffEngine
from .backends.duckdb_backend import DuckDBBackend
from .cache.memory_cache import MemoryCache
from .cache.redis_cache import RedisCache
from .types.pivot_spec import PivotSpec
from pivot_engine.streaming.streaming_processor import StreamAggregationProcessor, IncrementalMaterializedViewManager
from pivot_engine.hierarchical_scroll_manager import HierarchicalVirtualScrollManager
from pivot_engine.progressive_loader import ProgressiveDataLoader
from pivot_engine.cdc.cdc_manager import PivotCDCManager
from pivot_engine.materialized_hierarchy_manager import MaterializedHierarchyManager, IntelligentPrefetchManager
from pivot_engine.pruning_manager import HierarchyPruningManager, ProgressiveHierarchicalLoader


def sanitize_column_name(value: str) -> str:
    """Sanitize column value for use in SQL identifier"""
    import re
    if not value:
        return "null"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(value))
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized[:63]


class ScalablePivotController:
    """
    Scalable controller for pivot operations with advanced features for large datasets.
    Coordinates: Enhanced Planner -> Streaming Processor -> Incremental Views -> Diff Engine -> Cache -> Backend
    """
    
    def __init__(
        self,
        backend_uri: str = ":memory:",
        cache: Union[str, Any] = "memory",
        planner: Optional[Any] = None,
        planner_name: str = "ibis",
        enable_tiles: bool = True,
        enable_delta: bool = True,
        enable_streaming: bool = True,
        enable_incremental_views: bool = True,
        tile_size: int = 100,
        cache_ttl: int = 300,
        **cache_options: Any
    ):
        self.enable_arrow = True
        if backend_uri == ":memory:":
            backend_uri = ":memory:shared_pivot_db"

        self.backend = DuckDBBackend(uri=backend_uri)

        if isinstance(cache, str):
            if cache == "redis":
                self.cache = RedisCache(**cache_options)
            elif cache == "memory":
                self.cache = MemoryCache(ttl=cache_ttl)
            else:
                raise ValueError(f"Unknown cache type: {cache}")
        else:
            self.cache = cache or MemoryCache(ttl=cache_ttl)

        if planner:
            self.planner = planner
        elif planner_name == "ibis":
            try:
                import ibis
                # Support different Ibis backends based on URI
                if backend_uri.startswith("postgres://"):
                    from urllib.parse import urlparse
                    parsed = urlparse(backend_uri)
                    ibis_con = ibis.postgres.connect(
                        host=parsed.hostname,
                        port=parsed.port,
                        user=parsed.username,
                        password=parsed.password,
                        database=parsed.path[1:]  # Remove leading slash
                    )
                elif backend_uri.startswith("mysql://"):
                    from urllib.parse import urlparse
                    parsed = urlparse(backend_uri)
                    ibis_con = ibis.mysql.connect(
                        host=parsed.hostname,
                        port=parsed.port or 3306,
                        user=parsed.username,
                        password=parsed.password,
                        database=parsed.path[1:]
                    )
                elif backend_uri.startswith("bigquery://"):
                    ibis_con = ibis.bigquery.connect()
                elif backend_uri.startswith("snowflake://"):
                    from urllib.parse import urlparse
                    parsed = urlparse(backend_uri)
                    ibis_con = ibis.snowflake.connect(
                        user=parsed.username,
                        password=parsed.password,
                        account=parsed.hostname,
                    )
                elif backend_uri.startswith("sqlite://"):
                    db_path = backend_uri.replace("sqlite://", "")
                    ibis_con = ibis.sqlite.connect(db_path)
                else:
                    # Default to DuckDB
                    ibis_con = ibis.duckdb.connect(backend_uri)

                self.planner = IbisPlanner(con=ibis_con)
            except ImportError:
                self.planner = SQLPlanner(dialect="duckdb")
            except Exception as e:
                print(f"Could not connect to database backend: {e}, falling back to SQLPlanner")
                self.planner = SQLPlanner(dialect="duckdb")
        else:
            self.planner = SQLPlanner(dialect="duckdb")

        # Enhanced components for scalability
        self.diff_engine = QueryDiffEngine(
            cache=self.cache,
            default_ttl=cache_ttl,
            tile_size=tile_size,
            enable_tiles=enable_tiles,
            enable_delta_updates=enable_delta
        )
        
        self.tree_manager = TreeExpansionManager(self)
        
        # Scalability features (already set earlier)
        self.enable_streaming = enable_streaming
        if enable_streaming:
            self.streaming_processor = StreamAggregationProcessor()

        self.enable_incremental_views = enable_incremental_views
        if enable_incremental_views:
            self.incremental_view_manager = IncrementalMaterializedViewManager(self.backend)

        # Advanced hierarchical managers
        self.materialized_hierarchy_manager = MaterializedHierarchyManager(self.backend, self.cache)

        # Performance managers
        self.virtual_scroll_manager = HierarchicalVirtualScrollManager(self.planner, self.cache, self.materialized_hierarchy_manager)
        self.progressive_loader = ProgressiveDataLoader(self.backend, self.cache)


        # Initialize real pattern analyzer for intelligent prefetching
        from pivot_engine.materialized_hierarchy_manager import UserPatternAnalyzer

        self.intelligent_prefetch_manager = IntelligentPrefetchManager(
            session_tracker=None,  # Would be injected
            pattern_analyzer=UserPatternAnalyzer(cache=self.cache),  # Real pattern analyzer
            backend=self.backend,
            cache=self.cache
        )
        self.pruning_manager = HierarchyPruningManager(self.backend)
        self.progressive_hierarchy_loader = ProgressiveHierarchicalLoader(
            self.backend, self.cache, self.pruning_manager
        )

        # CDC for real-time updates
        self.cdc_manager = None  # Will be set via setup_cdc method
        
        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    async def setup_cdc(self, table_name: str, change_stream):
        """Setup CDC for real-time tracking of data changes"""
        self.cdc_manager = PivotCDCManager(self.backend, change_stream)

        await self.cdc_manager.setup_cdc(table_name)

        # Register materialized view manager to receive change notifications
        self.cdc_manager.register_materialized_view_manager(table_name, self.incremental_view_manager)

        # Start tracking changes in the background
        asyncio.create_task(self.cdc_manager.track_changes(table_name))

        return self.cdc_manager

    async def run_streaming_aggregation(self, spec: PivotSpec):
        """Run streaming aggregation for real-time results"""
        if not self.enable_streaming:
            raise ValueError("Streaming aggregation is not enabled")
        
        job_id = await self.streaming_processor.create_real_time_aggregation_job(spec)
        return {"job_id": job_id, "status": "created"}

    async def create_incremental_view(self, spec: PivotSpec):
        """Create incremental materialized view"""
        if not self.enable_incremental_views:
            raise ValueError("Incremental views are not enabled")
        
        view_name = await self.incremental_view_manager.create_incremental_view(spec)
        return {"view_name": view_name, "status": "created"}

    def run_hierarchical_pivot(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Standard hierarchical pivot"""
        return self.tree_manager.run_hierarchical_pivot(spec)

    def toggle_expansion(self, spec_hash: str, path: List[str]) -> Dict[str, Any]:
        """Toggle expansion of a hierarchical path"""
        return self.tree_manager.toggle_expansion(spec_hash, path)

    def run_virtual_scroll_hierarchical(self, spec: PivotSpec, start_row: int, end_row: int, expanded_paths: List[List[str]]):
        """Run hierarchical pivot with virtual scrolling for large datasets"""
        result = self.virtual_scroll_manager.get_visible_rows_hierarchical(
            spec, start_row, end_row, expanded_paths
        )
        return result

    async def run_progressive_load(self, spec: PivotSpec, chunk_callback: Optional[Callable] = None):
        """Run progressive data loading for large datasets"""
        result = await self.progressive_loader.load_progressive_chunks(spec, chunk_callback)
        return result

    async def run_hierarchical_progressive(self, spec: PivotSpec, expanded_paths: List[List[str]], level_callback: Optional[Callable] = None):
        """Run hierarchical data loading progressively by levels"""
        result = await self.progressive_loader.load_hierarchical_progressive(spec, expanded_paths, level_callback)
        return result

    def run_pivot(
        self,
        spec: Any,
        return_format: str = "arrow",
        force_refresh: bool = False
    ) -> Union[Dict[str, Any], pa.Table]:
        """Execute a pivot query with all scalability features"""
        self._request_count += 1
        spec = self._normalize_spec(spec)
        plan = self.planner.plan(spec)
        metadata = plan.get("metadata", {})

        if metadata.get("needs_column_discovery"):
            result_table = self._execute_topn_pivot(spec, plan, force_refresh)
        else:
            result_table = self._execute_standard_pivot(spec, plan, force_refresh)

        # Final conversion based on requested format
        if return_format == "dict":
            return self._convert_table_to_dict(result_table, spec)

        return result_table

    def _execute_standard_pivot(self, spec: Any, plan: Dict[str, Any], force_refresh: bool) -> pa.Table:
        """Execute standard pivot with all scalability optimizations"""
        queries_to_run, strategy = self.diff_engine.plan(plan, spec, force_refresh=force_refresh)

        self._cache_hits += strategy.get("cache_hits", 0)
        self._cache_misses += len(queries_to_run)

        results = [self.backend.execute(query) for query in queries_to_run]

        # Get the main aggregation result if available
        main_result = results[0] if results else pa.table({}) if pa is not None else None

        # Compute totals using Arrow operations if needed
        metadata = plan.get("metadata", {})
        if metadata.get("needs_totals", False) and main_result is not None and main_result.num_rows > 0:
            # Calculate totals from the main result using Arrow compute
            main_result = self._compute_totals_arrow(main_result, metadata)

        final_table = self.diff_engine.merge_and_finalize([main_result] if main_result is not None else [], plan, spec, strategy)

        return main_result if main_result is not None else pa.table({}) if pa is not None else None

    def _execute_topn_pivot(self, spec: Any, plan: Dict[str, Any], force_refresh: bool) -> pa.Table:
        """Execute top-N pivot"""
        queries = plan.get("queries", [])
        col_query = queries[0]
        col_cache_key = self._cache_key_for_query(col_query, spec)
        cached_cols_table = self.cache.get(col_cache_key) if not force_refresh else None

        if cached_cols_table:
            column_values = cached_cols_table.column("_col_key").to_pylist()
            self._cache_hits += 1
        else:
            col_results_table = self.backend.execute(col_query)
            column_values = col_results_table.column("_col_key").to_pylist()
            self.cache.set(col_cache_key, col_results_table)
            self._cache_misses += 1

        pivot_query = self.planner.build_pivot_query_from_columns(spec, column_values)
        pivot_cache_key = self._cache_key_for_query(pivot_query, spec)
        cached_pivot_table = self.cache.get(pivot_cache_key) if not force_refresh else None

        if cached_pivot_table:
            result_table = cached_pivot_table
        else:
            pivot_results_table = self.backend.execute(pivot_query)
            self.cache.set(pivot_cache_key, pivot_results_table)
            result_table = pivot_results_table

        # Compute totals using Arrow operations if needed
        metadata = plan.get("metadata", {})
        if metadata.get("needs_totals", False) and result_table is not None and result_table.num_rows > 0:
            result_table = self._compute_totals_arrow(result_table, metadata)

        return result_table

    def load_data_from_arrow(
        self,
        table_name: str,
        arrow_table: pa.Table,
        register_checkpoint: bool = True
    ):
        """Load data from Arrow table with CDC registration"""
        self.backend.create_table_from_arrow(table_name, arrow_table)
        if register_checkpoint:
            self.register_delta_checkpoint(table_name, timestamp=time.time())
        
        # If CDC is enabled, register the table for change tracking
        if self.cdc_manager:
            asyncio.create_task(self.cdc_manager.setup_cdc(table_name))

    def register_delta_checkpoint(self, table: str, timestamp: float = None, max_id: Optional[int] = None, incremental_field: str = "updated_at"):
        """Register a delta checkpoint for incremental updates"""
        timestamp = timestamp or time.time()
        self.diff_engine.register_delta_checkpoint(table, timestamp, max_id, incremental_field)

    def _normalize_spec(self, spec: Any) -> Any:
        """Normalize the pivot spec"""
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
        if table.num_rows == spec.limit:
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

        # Calculate totals for each aggregation column
        total_values = {}
        for col_name in table.column_names:
            if col_name in agg_aliases:
                col_array = table.column(col_name)

                # For aggregated values, use SUM to calculate grand totals
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

    def _cache_key_for_query(self, query: Dict[str, Any], spec: Any) -> str:
        """Generate cache key for a query"""
        import json
        import hashlib
        spec_dict = spec.to_dict() if hasattr(spec, 'to_dict') else spec
        items = {
            "sql": query.get("sql"),
            "params": tuple(query.get("params", [])),
            "spec_hash": hashlib.sha256(json.dumps(spec_dict, sort_keys=True, default=str).encode()).hexdigest()[:16]
        }
        key_str = json.dumps(items, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:32]
        return f"pivot:query:{key_hash}"

    def run_hierarchical_pivot_batch_load(
        self,
        spec: Dict[str, Any],
        expanded_paths: List[List[str]],
        max_levels: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run batch loading of multiple levels of the hierarchy"""
        return self.tree_manager._load_multiple_levels_batch(spec, expanded_paths, max_levels)

    def run_materialized_hierarchy(self, spec: PivotSpec):
        """Run hierarchical pivot using materialized rollups"""
        self.materialized_hierarchy_manager.create_materialized_hierarchy(spec)
        # Return empty result as hierarchy is now materialized
        return {"status": "materialized", "hierarchy_name": f"hierarchy_{spec.table}_{hash(str(spec.to_dict()))}"}

    async def run_intelligent_prefetch(self, spec: PivotSpec, user_session: Dict[str, Any], expanded_paths: List[List[str]]):
        """Run intelligent prefetching based on user behavior patterns"""
        prefetch_paths = await self.intelligent_prefetch_manager.determine_prefetch_strategy(
            user_session, spec, expanded_paths
        )
        return {"prefetch_paths": prefetch_paths, "status": "prefetching"}

    def run_progressive_hierarchical_load(self, spec: PivotSpec, expanded_paths: List[List[str]],
                                              user_preferences: Optional[Dict[str, Any]] = None,
                                              progress_callback: Optional[Callable] = None):
        """Run progressive hierarchical loading with pruning"""
        result = self.progressive_hierarchy_loader.load_progressive_hierarchy(
            spec, expanded_paths, user_preferences, progress_callback
        )
        return result

    def run_pruned_hierarchical_pivot(self, spec: PivotSpec, expanded_paths: List[List[str]],
                                          user_preferences: Dict[str, Any]):
        """Run hierarchical pivot with intelligent pruning"""
        # Get the hierarchical data
        hierarchy_data = self.materialized_hierarchy_manager.get_optimized_hierarchical_data(
            spec, expanded_paths
        )

        # Apply pruning
        pruned_data = self.pruning_manager.apply_pruning_strategy(
            hierarchy_data, user_preferences
        )

        return {"data": pruned_data, "pruning_applied": True}

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

    def clear_cache(self):
        """Clear all cached queries to force fresh data retrieval"""
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()

    def close(self):
        """Close any resources held by the controller"""
        if hasattr(self, 'backend') and hasattr(self.backend, 'close'):
            self.backend.close()