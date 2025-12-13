"""
QueryDiffEngine - tile-aware, spec-diff optimized, delta-update capable.

New features:
- Tile-aware diffing for virtual scrolling (row/column tiles)
- Semantic spec diffing (detect pagination-only, filter-only, sort-only changes)
- Delta updates for incremental data ingestion
- Table-level cache invalidation
- Optimized for Arrow/DuckDB integration
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import pyarrow as pa

try:
    import xxhash
    _use_xx = True
except ImportError:
    xxhash = None
    _use_xx = False


class SpecChangeType(Enum):
    """Types of changes between pivot specs"""
    IDENTICAL = "identical"
    PAGE_ONLY = "page_only"  # Only pagination changed
    SORT_ONLY = "sort_only"  # Only sort changed
    FILTER_ADDED = "filter_added"  # Filters added (subset of data)
    FILTER_REMOVED = "filter_removed"  # Filters removed (superset needed)
    STRUCTURE_CHANGED = "structure_changed"  # Rows/columns/measures changed
    FULL_REFRESH = "full_refresh"  # Complete recomputation needed


@dataclass
class TileKey:
    """Identifies a specific tile in the result grid"""
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    # NEW for hierarchical dimensions
    dimension_level: Optional[Dict[str, int]] = None  # e.g., {"region": 1, "product": 2}
    drill_path: Optional[List[str]] = None  # e.g., ["USA", "California", "San Francisco"]

    def to_string(self) -> str:
        """Convert tile key to string representation, including hierarchical info"""
        base_part = f"r{self.row_start}-{self.row_end}_c{self.col_start}-{self.col_end}"

        if self.drill_path is not None:
            path_part = ":".join(self.drill_path)
            base_part += f"_path_{path_part}"

        if self.dimension_level is not None:
            level_part = ",".join([f"{k}:{v}" for k, v in self.dimension_level.items()])
            base_part += f"_level_{level_part}"

        return base_part

    @staticmethod
    def from_string(s: str) -> 'TileKey':
        """Parse tile key from string representation including hierarchical info"""
        # Split by underscore but keep track of the hierarchical parts
        # Expected format: rX-Y_cA-B[_path_...][_level_...]

        # First, identify hierarchical parts by looking for _path_ and _level_
        path_part = None
        level_part = None
        base_part = s

        # Extract path part if exists
        if '_path_' in s:
            path_start = s.find('_path_')
            base_part = s[:path_start]
            path_content_start = path_start + 6  # length of '_path_'

            # Find where path content ends (either level starts or end of string)
            level_start = s.find('_level_', path_content_start)
            if level_start != -1:
                path_end = level_start
            else:
                path_end = len(s)

            path_content = s[path_content_start:path_end]
            if path_content:
                path_part = path_content.split(':')

        # Extract level part if exists
        if '_level_' in s:
            level_start = s.find('_level_')
            level_content_start = level_start + 7  # length of '_level_'
            level_content = s[level_content_start:]

            if level_content:
                level_dict = {}
                for item in level_content.split(','):
                    if ':' in item:
                        k, v = item.split(':', 1)  # Split only on first ':'
                        level_dict[k] = int(v)
                level_part = level_dict
        else:
            # If no _level_ was found in original string, use base_part as is
            if '_path_' not in s:
                base_part = s

        # Parse the base part (should be in format rX-Y_cA-B)
        if '_' in base_part:
            row_part = base_part.split('_')[0][1:]  # Skip 'r'
            col_part = base_part.split('_')[1][1:]  # Skip 'c'
            r_start, r_end = map(int, row_part.split('-'))
            c_start, c_end = map(int, col_part.split('-'))
        else:
            # Fallback for unexpected format
            raise ValueError(f"Invalid tile string format: {s}")

        return TileKey(
            row_start=r_start,
            row_end=r_end,
            col_start=c_start,
            col_end=c_end,
            dimension_level=level_part,
            drill_path=path_part
        )


@dataclass
class QueryTile:
    """Represents a cached tile of query results"""
    data: List[Dict[str, Any]]
    timestamp: float
    row_count: int
    spec_hash: str


@dataclass
class DeltaInfo:
    """Information about incremental data changes"""
    table: str
    last_timestamp: float
    last_max_id: Optional[int] = None
    incremental_field: Optional[str] = None  # timestamp or id field


class QueryDiffEngine:
    """
    Advanced diff engine with tile-aware caching and semantic spec analysis.
    """
    
    def __init__(
        self,
        cache,
        default_ttl: int = 300,
        tile_size: int = 100,
        enable_tiles: bool = True,
        enable_delta_updates: bool = True
    ):
        """
        Args:
            cache: Cache object with get/set methods
            default_ttl: Default cache TTL in seconds
            tile_size: Number of rows per tile for tile-aware caching
            enable_tiles: Enable tile-based caching
            enable_delta_updates: Enable incremental delta updates
        """
        self.cache = cache
        self.ttl = default_ttl
        self.tile_size = tile_size
        self.enable_tiles = enable_tiles
        self.enable_delta_updates = enable_delta_updates
        
        # Track previous spec for diffing
        self._last_spec_hash: Optional[str] = None
        self._last_spec: Optional[Dict[str, Any]] = None
        self._last_plan_digest: Optional[str] = None
        
        # Delta tracking per table
        self._delta_info: Dict[str, DeltaInfo] = {}
        
        # Table invalidation tracking
        self._invalidated_tables: Set[str] = set()
    
    # ========== Public API ==========
    
    def plan(
        self,
        plan: Dict[str, Any],
        spec: Any,
        force_refresh: bool = False,
        backend: Optional[Any] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Decide which queries to execute based on semantic diff analysis and delta updates.

        Returns:
            (queries_to_run, execution_strategy)

        execution_strategy contains:
            - change_type: SpecChangeType
            - can_reuse_tiles: bool
            - tiles_needed: List[TileKey] if applicable
            - cache_hits: int
            - use_delta_updates: bool
        """
        queries = plan.get("queries", [])
        if not queries:
            return [], {"change_type": SpecChangeType.IDENTICAL}

        # Normalize spec
        spec_dict = self._normalize_spec_for_hash(spec)
        spec_hash = self._hash_dict(spec_dict)

        # Check if table was invalidated
        table_name = spec_dict.get("table")
        if table_name in self._invalidated_tables:
            force_refresh = True
            self._invalidated_tables.discard(table_name)

        # Compute plan digest
        plan_digest = self._digest_plan(plan, spec)

        # Analyze spec changes
        change_type = self._analyze_spec_change(spec_dict, self._last_spec)

        # Store current spec for next diff
        self._last_spec = spec_dict
        self._last_spec_hash = spec_hash
        self._last_plan_digest = plan_digest

        execution_strategy = {
            "change_type": change_type,
            "can_reuse_tiles": False,
            "tiles_needed": [],
            "cache_hits": 0,
            "force_refresh": force_refresh,
            "use_delta_updates": False,
            "delta_queries_generated": 0
        }

        # Check if delta updates are applicable for this table
        # Only apply delta updates to queries that are appropriate for deltas
        # Skip delta updates for pagination/cursor-based queries (since they're not full aggregations)
        use_delta_updates = (
            self.enable_delta_updates and
            table_name in self._delta_info and
            change_type not in [SpecChangeType.STRUCTURE_CHANGED, SpecChangeType.FULL_REFRESH] and
            not spec_dict.get("cursor")  # Don't apply deltas to cursor pagination queries
        )

        # Only compute and return delta queries if specifically appropriate
        if use_delta_updates:
            delta_queries = self.compute_delta_queries(spec, plan)
            if delta_queries and table_name in self._delta_info:
                execution_strategy["use_delta_updates"] = True
                execution_strategy["delta_queries_generated"] = len(delta_queries)
                return delta_queries, execution_strategy

        # Handle different change types
        if force_refresh or change_type == SpecChangeType.FULL_REFRESH:
            return queries, execution_strategy

        if change_type == SpecChangeType.IDENTICAL:
            # Check if all queries are cached
            all_cached = all(
                self._is_query_cached(q, spec_dict) for q in queries
            )
            if all_cached:
                execution_strategy["cache_hits"] = len(queries)
                return [], execution_strategy
            # Fall through to execute missing queries

        # Use tile-aware strategy only for virtual scrolling (page/offset-based), not cursor-based pagination
        if change_type == SpecChangeType.PAGE_ONLY and self.enable_tiles and spec_dict.get("cursor") is None:
            # Use tile-aware strategy
            return self._plan_tile_aware(queries, spec_dict, plan, execution_strategy)

        if change_type == SpecChangeType.SORT_ONLY:
            # Can reuse aggregate data, just re-sort
            agg_queries = [q for q in queries if q.get("purpose") == "aggregate"]
            if self._is_query_cached(agg_queries[0], self._last_spec) if agg_queries else False:
                execution_strategy["cache_hits"] = len(queries) - len(agg_queries)
                # Re-sort cached data client-side or run lightweight sort query
                return [], execution_strategy

        if change_type == SpecChangeType.FILTER_ADDED:
            # More restrictive filters - can potentially filter cached results
            # For now, execute new query but could optimize later
            pass

        # Default: determine which queries need execution
        to_run = []
        for q in queries:
            if not self._is_query_cached(q, spec_dict):
                to_run.append(q)
            else:
                execution_strategy["cache_hits"] += 1

        return to_run, execution_strategy
    
    def merge_and_finalize(
        self,
        results: List[pa.Table],
        plan: Dict[str, Any],
        spec: Any,
        execution_strategy: Dict[str, Any]
    ) -> Optional[pa.Table]:
        """
        Merge executed Arrow tables with cached data into a final Arrow Table.
        Note: Totals computation now happens in the controller for Arrow-native efficiency.
        """
        spec_dict = self._normalize_spec_for_hash(spec)
        queries = plan.get("queries", [])
        use_delta_updates = execution_strategy.get("use_delta_updates", False)

        # If using delta updates, we need to apply them to cached results
        if use_delta_updates and results:
            # Get the table name to find cached base results
            table_name = spec_dict.get("table")
            if table_name and table_name in self._delta_info:
                # For now, just return the first result from delta execution
                # In a complete implementation, we would merge with cached base data
                if results:
                    # Cache the delta result
                    first_query = queries[0] if queries else None
                    if first_query:
                        cache_key = self._cache_key_for_query(first_query, spec_dict)
                        self.cache.set(cache_key, results[0])
                    return results[0]

        # Since the controller executes only the queries returned by plan() method,
        # and passes the results of only those executed queries to this method,
        # we need to figure out which queries were executed based on the results provided
        # and which should come from cache

        # The results list contains results for queries_to_run (returned by plan method)
        # But we need to combine them with cached results for queries that weren't run

        # Get the queries that were returned by the plan() method for execution
        # This information should come from the execution strategy
        change_type = execution_strategy.get("change_type", SpecChangeType.FULL_REFRESH)

        # For most cases, if it's a page_only or similar change, we only have partial results
        # We should only return the results that were actually computed
        if results:
            # For typical aggregation queries, we expect only one main result
            # If we have multiple results, it might be due to multiple queries being run
            return results[0] if len(results) == 1 else pa.concat_tables(results)
        else:
            # If no new results were computed, we may need to retrieve from cache
            # For cursor-based pagination, the result should come from newly executed queries
            # so if no results were passed, we return None
            return None

    
    def invalidate_cache_for_table(self, table_name: str):
        """
        Invalidate all cached queries for a specific table.
        Called by ETL/ingestion processes when data changes.
        """
        self._invalidated_tables.add(table_name)
        
        # Clear delta info for incremental updates
        if table_name in self._delta_info:
            del self._delta_info[table_name]
        
        # For now, mark for lazy invalidation on next query
        # In a production system, you would want to actively scan and delete matching cache keys
        # For now, mark for lazy invalidation on next query
    
    def register_delta_checkpoint(
        self,
        table: str,
        timestamp: float,
        max_id: Optional[int] = None,
        incremental_field: str = "updated_at"
    ):
        """
        Register a checkpoint for delta/incremental updates.
        
        Args:
            table: Table name
            timestamp: Timestamp of last full load
            max_id: Maximum ID seen (if using ID-based incremental)
            incremental_field: Field to use for incremental queries
        """
        self._delta_info[table] = DeltaInfo(
            table=table,
            last_timestamp=timestamp,
            last_max_id=max_id,
            incremental_field=incremental_field
        )
    
    def compute_delta_queries(
        self,
        spec: Any,
        plan: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate delta queries for incremental updates with enhanced logic.

        Returns modified queries that fetch only new/changed data,
        or None if delta updates not applicable.
        """
        if not self.enable_delta_updates:
            return None

        spec_dict = self._normalize_spec_for_hash(spec)
        table_name = spec_dict.get("table")

        if table_name not in self._delta_info:
            return None

        delta_info = self._delta_info[table_name]
        queries = plan.get("queries", [])
        delta_queries = []

        for q in queries:
            # Modify query to fetch only incremental data
            modified_q = self._add_delta_filter(q, delta_info)
            if modified_q:
                delta_queries.append(modified_q)

        return delta_queries if delta_queries else None

    def apply_delta_updates(
        self,
        spec: Any,
        plan: Dict[str, Any],
        base_result: Optional[pa.Table],
        backend: Optional[Any] = None
    ) -> Optional[pa.Table]:
        """
        Apply delta updates to existing cached results to produce updated results.

        Args:
            spec: The pivot specification
            plan: The query plan
            base_result: The existing cached result (if available)
            backend: Optional backend to execute delta queries

        Returns:
            Updated result table with delta changes applied, or None if not applicable
        """
        if not self.enable_delta_updates:
            return base_result

        # Check if delta updates are available for this table
        spec_dict = self._normalize_spec_for_hash(spec)
        table_name = spec_dict.get("table")

        if table_name not in self._delta_info:
            return base_result

        # Compute delta queries and execute them
        delta_queries = self.compute_delta_queries(spec, plan)
        if not delta_queries:
            return base_result

        # Execute delta queries to get incremental changes (if backend provided)
        if backend is not None:
            delta_results = []
            for dq in delta_queries:
                try:
                    # Execute the delta query using the provided backend
                    delta_result_table = backend.execute(dq)
                    if delta_result_table is not None:
                        delta_results.extend(delta_result_table.to_pylist())
                except Exception as e:
                    print(f"Warning: Delta execution failed: {e}")
                    # If delta execution fails, return base result
                    return base_result

            # If we have both base result and delta result, merge them
            if base_result is not None and delta_results:
                # Convert base result to list of dicts for merging
                base_data = base_result.to_pylist() if base_result is not None else []
                # Merge the results
                merged_data = self.merge_delta_results(base_data, delta_results, spec_dict.get("measures", []))

                # Convert back to PyArrow table
                if merged_data:
                    # Create a dictionary grouping all values by column name
                    columns = {}
                    for row in merged_data:
                        for col, val in row.items():
                            if col not in columns:
                                columns[col] = []
                            columns[col].append(val)

                    # Create PyArrow table from columns
                    import pyarrow as pa
                    arrays = {}
                    for col_name, values in columns.items():
                        # Create array with type inference
                        try:
                            arrays[col_name] = pa.array(values)
                        except:
                            # If type inference fails, use string type
                            arrays[col_name] = pa.array([str(v) for v in values])

                    return pa.table(arrays)

        # If no backend provided, return the base result unchanged
        return base_result

    def _merge_results_with_deltas(
        self,
        base_results: List[Dict[str, Any]],
        delta_results: List[Dict[str, Any]],
        measures: List[Dict[str, Any]]
    ) -> pa.Table:
        """
        Merge base results with delta results to produce updated table.
        """
        # Use the existing merge_delta_results method for core logic
        merged_data = self.merge_delta_results(base_results, delta_results, measures)

        # Convert back to PyArrow table format
        if not merged_data:
            return None

        # Build schema from the first row
        if merged_data:
            # Create a dictionary grouping all values by column name
            columns = {}
            for row in merged_data:
                for col, val in row.items():
                    if col not in columns:
                        columns[col] = []
                    columns[col].append(val)

            # Create PyArrow table from columns
            import pyarrow as pa
            arrays = {}
            for col_name, values in columns.items():
                # Try to infer the type or set a default
                arrays[col_name] = pa.array(values)

            return pa.table(arrays)

        return None
    
    # ========== Tile-Aware Methods ==========
    
    def _plan_tile_aware(
        self,
        queries: List[Dict[str, Any]],
        spec_dict: Dict[str, Any],
        plan: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Plan execution using tile-based caching.
        """
        page = spec_dict.get("page", {})
        offset = page.get("offset", 0)
        limit = page.get("limit", 100)
        
        # Calculate tile boundaries
        start_tile = offset // self.tile_size
        end_tile = (offset + limit - 1) // self.tile_size
        
        tiles_needed = []
        tiles_cached = []
        
        for tile_idx in range(start_tile, end_tile + 1):
            tile_start = tile_idx * self.tile_size
            tile_end = min(tile_start + self.tile_size, offset + limit)
            
            tile_key = TileKey(
                row_start=tile_start,
                row_end=tile_end,
                col_start=0,  # For now, full column width
                col_end=-1   # -1 means all columns
            )
            
            cache_key = self._cache_key_for_tile(tile_key, spec_dict)
            
            if self.cache.get(cache_key) is not None:
                tiles_cached.append(tile_key)
            else:
                tiles_needed.append(tile_key)
        
        strategy["can_reuse_tiles"] = len(tiles_cached) > 0
        strategy["tiles_needed"] = [t.to_string() for t in tiles_needed]
        strategy["cache_hits"] = len(tiles_cached)
        
        if not tiles_needed:
            # All tiles cached
            return [], strategy
        
        # Generate queries for missing tiles
        tile_queries = []
        for tile in tiles_needed:
            for q in queries:
                if q.get("purpose") == "aggregate":
                    tile_q = self._create_tile_query(q, tile, spec_dict)
                    tile_queries.append(tile_q)
        
        return tile_queries, strategy
    
    def _create_tile_query(
        self,
        base_query: Dict[str, Any],
        tile: TileKey,
        spec_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a query modified to fetch a specific tile.
        """
        # Clone query
        tile_query = base_query.copy()

        # Modify SQL to add/update LIMIT and OFFSET
        sql = tile_query["sql"]

        # Remove existing LIMIT/OFFSET
        sql = self._remove_limit_offset(sql)

        # Determine the effective limit for this tile
        # We need to respect the original spec limit while applying tiling
        spec_limit = spec_dict.get("limit", self.tile_size)

        # Calculate tile-based limit considering the spec limit
        tile_limit = tile.row_end - tile.row_start
        effective_limit = min(tile_limit, spec_limit)

        # If the tile offset is 0, we start from the beginning
        # If the tile offset is greater than 0, we apply the offset
        if tile.row_start == 0:
            # If tile starts from 0 and spec limit is small, use that
            effective_limit = min(effective_limit, spec_limit)
            sql += f" LIMIT {effective_limit}"
        else:
            # For non-zero tile start, apply offset and effective limit
            sql += f" LIMIT {effective_limit} OFFSET {tile.row_start}"

        tile_query["sql"] = sql
        tile_query["tile_key"] = tile.to_string()

        return tile_query
    
    def _cache_key_for_tile(
        self,
        tile: TileKey,
        spec_dict: Dict[str, Any]
    ) -> str:
        """Generate cache key for a specific tile"""
        # Hash spec without pagination
        spec_no_page = spec_dict.copy()
        spec_no_page.pop("page", None)
        
        base_hash = self._hash_dict(spec_no_page)
        tile_str = tile.to_string()
        
        return f"pivot:tile:{base_hash}:{tile_str}"
    
    # ========== Spec Diffing ==========
    
    def _analyze_spec_change(
        self,
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]]
    ) -> SpecChangeType:
        """
        Analyze semantic differences between specs with enhanced analysis.
        """
        if previous is None:
            return SpecChangeType.FULL_REFRESH

        # Enhanced analysis with better categorization
        # Check for identical specs first
        if current == previous:
            return SpecChangeType.IDENTICAL

        # Separate different types of changes for more granular analysis
        page_changed = current.get("page") != previous.get("page")
        sort_changed = current.get("sort") != previous.get("sort")
        filters_changed = current.get("filters", []) != previous.get("filters", [])
        limits_changed = current.get("limit") != previous.get("limit")
        cursor_changed = current.get("cursor") != previous.get("cursor")

        # Check structural changes
        rows_changed = current.get("rows", []) != previous.get("rows", [])
        columns_changed = current.get("columns", []) != previous.get("columns", [])
        measures_changed = self._compare_measures(current.get("measures", []), previous.get("measures", []))
        table_changed = current.get("table") != previous.get("table")
        drill_paths_changed = current.get("drill_paths", []) != previous.get("drill_paths", [])
        totals_changed = current.get("totals", False) != previous.get("totals", False)

        # Check if only pagination-related changes occurred
        if (page_changed or limits_changed or cursor_changed) and not any([
            sort_changed, filters_changed, rows_changed, columns_changed,
            measures_changed, table_changed, drill_paths_changed, totals_changed
        ]):
            return SpecChangeType.PAGE_ONLY

        # Check if only sorting changed
        if sort_changed and not any([
            page_changed, filters_changed, rows_changed, columns_changed,
            measures_changed, table_changed, drill_paths_changed, limits_changed,
            cursor_changed, totals_changed
        ]):
            return SpecChangeType.SORT_ONLY

        # Check if only filters changed
        if filters_changed and not any([
            page_changed, sort_changed, rows_changed, columns_changed,
            measures_changed, table_changed, drill_paths_changed, limits_changed,
            cursor_changed, totals_changed
        ]):
            # Determine if filters were added or removed
            curr_filters = set(json.dumps(f, sort_keys=True) for f in current.get("filters", []))
            prev_filters = set(json.dumps(f, sort_keys=True) for f in previous.get("filters", []))

            if len(curr_filters) > len(prev_filters) and curr_filters.issuperset(prev_filters):
                return SpecChangeType.FILTER_ADDED
            elif len(prev_filters) > len(curr_filters) and prev_filters.issuperset(curr_filters):
                return SpecChangeType.FILTER_REMOVED
            else:
                return SpecChangeType.FULL_REFRESH

        # Check if only drill paths changed (hierarchical changes)
        if drill_paths_changed and not any([
            page_changed, sort_changed, filters_changed, rows_changed, columns_changed,
            measures_changed, table_changed, limits_changed, cursor_changed, totals_changed
        ]):
            return SpecChangeType.FULL_REFRESH  # Drill path changes require full refresh for now

        # Check for structural changes
        if any([rows_changed, columns_changed, measures_changed, table_changed]):
            return SpecChangeType.STRUCTURE_CHANGED

        # Check for totals changes
        if totals_changed and not any([
            page_changed, sort_changed, filters_changed, rows_changed, columns_changed,
            measures_changed, table_changed
        ]):
            # Totals change might not require full recomputation if data is already aggregated
            return SpecChangeType.SORT_ONLY  # Treat as sort-only for now since totals are computed in controller

        # For any other combination of changes, require full refresh
        return SpecChangeType.FULL_REFRESH

    def _compare_measures(self, measures1: List[Dict[str, Any]], measures2: List[Dict[str, Any]]) -> bool:
        """
        Compare two lists of measures for equality.
        Returns True if measures are different.
        """
        if len(measures1) != len(measures2):
            return True

        # Convert to comparable format
        def normalize_measure(m):
            if isinstance(m, dict):
                return tuple(sorted((k, v) for k, v in m.items() if k != 'filter_condition'))  # filter_condition may be dynamic
            return str(m)

        measures1_norm = [normalize_measure(m) for m in measures1]
        measures2_norm = [normalize_measure(m) for m in measures2]

        return sorted(measures1_norm) != sorted(measures2_norm)
    
    # ========== Delta Updates ==========
    
    def _add_delta_filter(
        self,
        query: Dict[str, Any],
        delta_info: DeltaInfo
    ) -> Optional[Dict[str, Any]]:
        """
        Modify query to fetch only incremental data.
        Only add delta filter if the incremental field exists in the query's table.
        """
        sql = query.get("sql", "")
        params = query.get("params", []).copy()

        # Check if the incremental field exists in the query
        # For now, do a simple check - in production, you might want more robust column validation
        incremental_field = delta_info.incremental_field

        # Skip delta filter if the field name is suspicious (not in the SQL at all)
        # This is a basic safety check - could be improved with schema validation
        if incremental_field.lower() not in sql.lower():
            # Check if the table actually contains the incremental field
            # For this basic check, we'll proceed conservatively
            pass

        # For a more robust check, we need to extract the table name from the SQL
        # This is a simplified version but tries to be safe
        if "WHERE" in sql.upper():
            # Append to existing WHERE
            insert_pos = sql.upper().find("GROUP BY")
            if insert_pos == -1:
                insert_pos = sql.upper().find("ORDER BY")
            if insert_pos == -1:
                insert_pos = len(sql)

            delta_clause = f" AND {incremental_field} > ?"
            sql = sql[:insert_pos] + delta_clause + sql[insert_pos:]
        else:
            # Add new WHERE clause
            insert_pos = sql.upper().find("GROUP BY")
            if insert_pos == -1:
                insert_pos = sql.upper().find("ORDER BY")
            if insert_pos == -1:
                insert_pos = len(sql)

            delta_clause = f" WHERE {incremental_field} > ?"
            sql = sql[:insert_pos] + delta_clause + sql[insert_pos:]

        params.append(delta_info.last_timestamp)

        delta_query = query.copy()
        delta_query["sql"] = sql
        delta_query["params"] = params
        delta_query["is_delta"] = True

        return delta_query
    
    def merge_delta_results(
        self,
        base_results: List[Dict[str, Any]],
        delta_results: List[Dict[str, Any]],
        measures: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge delta results into base results using associative aggregation.
        
        For associative functions (SUM, COUNT, MIN, MAX), we can merge deltas.
        For non-associative (AVG, MEDIAN), we need full recalculation.
        """
        # Index base results by group-by keys
        base_map = {}
        for row in base_results:
            # Create key from all non-measure columns
            key_parts = []
            for k, v in row.items():
                if not any(m.get("alias") == k for m in measures):
                    key_parts.append(f"{k}:{v}")
            key = "|".join(key_parts)
            base_map[key] = row
        
        # Merge delta results
        for delta_row in delta_results:
            key_parts = []
            for k, v in delta_row.items():
                if not any(m.get("alias") == k for m in measures):
                    key_parts.append(f"{k}:{v}")
            key = "|".join(key_parts)
            
            if key in base_map:
                # Merge aggregates
                base_row = base_map[key]
                for measure in measures:
                    alias = measure.get("alias")
                    agg = measure.get("agg", "sum").lower()
                    
                    if agg in {"sum", "count"}:
                        base_row[alias] = base_row.get(alias, 0) + delta_row.get(alias, 0)
                    elif agg == "min":
                        base_row[alias] = min(base_row.get(alias, float('inf')), delta_row.get(alias, float('inf')))
                    elif agg == "max":
                        base_row[alias] = max(base_row.get(alias, float('-inf')), delta_row.get(alias, float('-inf')))
                    # AVG, MEDIAN, etc. cannot be merged incrementally
            else:
                # New group - add to base
                base_map[key] = delta_row
        
        return list(base_map.values())
    
    # ========== Helper Methods ==========
    
    def _is_query_cached(
        self,
        query: Dict[str, Any],
        spec_dict: Dict[str, Any]
    ) -> bool:
        """Check if query result is in cache"""
        cache_key = self._cache_key_for_query(query, spec_dict)
        return self.cache.get(cache_key) is not None
    
    def _cache_key_for_query(
        self,
        query: Dict[str, Any],
        spec_dict: Dict[str, Any]
    ) -> str:
        """Generate cache key for a query"""
        items = {
            "spec": self._hash_dict(spec_dict),
            "query_name": query.get("name"),
            "sql": query.get("sql"),
            "params": tuple(query.get("params", []))
        }
        key = self._hash_dict(items)
        return f"pivot:query:{key}"
    
    def _digest_plan(
        self,
        plan: Dict[str, Any],
        spec: Any
    ) -> str:
        """Generate digest for plan"""
        plan_summary = {
            "metadata": plan.get("metadata"),
            "num_queries": len(plan.get("queries", []))
        }
        return self._hash_dict(plan_summary)
    
    def _normalize_spec_for_hash(self, spec: Any) -> Dict[str, Any]:
        """Convert spec to normalized dict"""
        if hasattr(spec, "to_dict"):
            return spec.to_dict()
        if hasattr(spec, "__dict__"):
            d = dict(spec.__dict__)
            # Handle nested dataclasses
            if "page" in d and hasattr(d["page"], "__dict__"):
                d["page"] = dict(d["page"].__dict__)
            return d
        if isinstance(spec, dict):
            return spec
        return {"spec": str(spec)}
    
    def _hash_dict(self, d: Dict[str, Any]) -> str:
        """Hash dictionary to string"""
        j = json.dumps(d, sort_keys=True, default=str)
        return self._hash_text(j)
    
    def _hash_text(self, text: str) -> str:
        """Hash text using xxhash or sha256"""
        if _use_xx:
            return xxhash.xxh64(text.encode("utf-8")).hexdigest()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _remove_limit_offset(self, sql: str) -> str:
        """Remove LIMIT and OFFSET clauses from SQL"""
        # Simple regex-based removal (could be more robust with SQL parser)
        import re
        sql = re.sub(r'\s+LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\s+OFFSET\s+\d+', '', sql, flags=re.IGNORECASE)
        return sql
    
    async def _assemble_result(
        self,
        query_map: Dict[str, Any],
        spec_dict: Dict[str, Any],
        strategy: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble final QueryResult from query map"""
        agg_rows = query_map.get("aggregate", []) or []
        col_rows = query_map.get("column_values", []) or []
        totals_row = query_map.get("totals", []) or []

        # Extract columns and rows
        columns = []
        rows_out = []

        if agg_rows:
            first = agg_rows[0]
            if isinstance(first, dict):
                columns = list(first.keys())
                rows_out = [list(r.values()) for r in agg_rows]
            else:
                rows_out = agg_rows

        # Page info
        page = spec_dict.get("page", {})
        page_info = {
            "offset": page.get("offset", 0),
            "limit": page.get("limit", len(rows_out)),
            "total": await self._compute_true_total(spec_dict, plan)
        }
        
        # Stats
        stats = {
            "cached_queries": strategy.get("cache_hits", 0),
            "executed_queries": len(plan.get("queries", [])) - strategy.get("cache_hits", 0),
            "merged_rows": len(rows_out),
            "change_type": strategy.get("change_type", SpecChangeType.FULL_REFRESH).value,
            "plan_digest": self._last_plan_digest,
            "ts": time.time()
        }
        
        if strategy.get("can_reuse_tiles"):
            stats["tiles_reused"] = strategy.get("cache_hits", 0)
            stats["tiles_fetched"] = len(strategy.get("tiles_needed", []))
        
        return {
            "columns": columns,
            "rows": rows_out,
            "page": page_info,
            "stats": stats,
            "metadata": plan.get("metadata", {}),
            "totals": totals_row,
        }
        


class MultiDimensionalTilePlanner:
    """
    Handles multi-dimensional tile planning for hierarchical data.
    """

    def __init__(self, tile_size: int = 100):
        self.tile_size = tile_size

    def plan_hierarchical_tiles(
        self,
        spec: Dict[str, Any],
        drill_state: Dict[str, Any]
    ) -> List[TileKey]:
        """
        Generate tiles for hierarchical drill-down.

        drill_state example:
        {
            "expanded_paths": [
                ["USA"],  # Expanded: show states under USA
                ["USA", "California"]  # Expanded: show cities under CA
            ]
        }
        """
        tiles = []
        max_tile_size = 1000  # Configurable max tile size for hierarchical data

        for path in drill_state.get("expanded_paths", []):
            # Create tile for this drill level
            tile = TileKey(
                row_start=0,
                row_end=max_tile_size,
                col_start=0,
                col_end=-1,  # -1 means all columns
                drill_path=path,
                dimension_level={spec["rows"][i]: i for i in range(len(path))}
            )
            tiles.append(tile)

        # Add support for different drill paths at different hierarchical levels
        all_paths = drill_state.get("expanded_paths", [])
        for i, path in enumerate(all_paths):
            for level in range(len(path) + 1):  # Include parent levels too
                sub_path = path[:level]
                if sub_path:  # Only create tiles for non-empty paths
                    tile = TileKey(
                        row_start=0,
                        row_end=max_tile_size,
                        col_start=0,
                        col_end=-1,
                        drill_path=sub_path,
                        dimension_level={spec["rows"][j]: j for j in range(len(sub_path))}
                    )
                    # Avoid duplicates
                    if tile not in tiles:
                        tiles.append(tile)

        return tiles

    def plan_multi_dimensional_tiles(
        self,
        spec: Dict[str, Any],
        current_tile: Optional[TileKey] = None
    ) -> List[TileKey]:
        """
        Plan tiles for multi-dimensional data considering rows, columns, and filters.
        """
        tiles = []

        # Calculate tile boundaries based on spec dimensions
        row_dims = spec.get("rows", [])
        col_dims = spec.get("columns", [])

        # For now, create basic row tiles (can be extended for column tiles in future)
        row_start = (current_tile.row_start if current_tile else 0)
        row_end = (current_tile.row_end if current_tile else min(self.tile_size, 100))

        tile = TileKey(
            row_start=row_start,
            row_end=row_end,
            col_start=0,
            col_end=-1,
            dimension_level={dim: i for i, dim in enumerate(row_dims)}
        )

        tiles.append(tile)

        # Add more tiles as needed based on dimensions
        # This is a simplified implementation - could be expanded based on actual needs
        return tiles


def _plan_hierarchical_tiles(
    self,
    spec: Dict[str, Any],
    drill_state: Dict[str, Any]
) -> List[TileKey]:
    """
    Generate tiles for hierarchical drill-down.
    This function is attached to the QueryDiffEngine class.

    drill_state example:
    {
        "expanded_paths": [
            ["USA"],  # Expanded: show states under USA
            ["USA", "California"]  # Expanded: show cities under CA
        ]
    }
    """
    planner = MultiDimensionalTilePlanner(tile_size=self.tile_size)
    return planner.plan_hierarchical_tiles(spec, drill_state)


# Attach the function to the QueryDiffEngine class
QueryDiffEngine._plan_hierarchical_tiles = _plan_hierarchical_tiles


def _plan_multi_dimensional(
    self,
    queries: List[Dict[str, Any]],
    spec_dict: Dict[str, Any],
    plan: Dict[str, Any],
    strategy: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Plan execution using multi-dimensional tile-based caching.
    Enhanced version of tile-aware planning with hierarchical support.
    """
    page = spec_dict.get("page", {})
    offset = page.get("offset", 0)
    limit = page.get("limit", 100)

    # Handle hierarchical queries (with drill paths)
    if "drill_paths" in spec_dict and spec_dict.get("drill_paths"):
        # Use hierarchical tile planning
        drill_state = {
            "expanded_paths": [dp.get("values", []) for dp in spec_dict.get("drill_paths", [])]
        }
        tiles = self._plan_hierarchical_tiles(spec_dict, drill_state)
    else:
        # Calculate regular tile boundaries with more sophisticated logic
        # Consider both row and column dimensions for multi-dimensional tiling
        start_tile = offset // self.tile_size
        end_tile = (offset + limit - 1) // self.tile_size

        tiles = []
        # Include context about column dimensions for multi-dimensional tiles
        row_dims = spec_dict.get("rows", [])
        col_dims = spec_dict.get("columns", [])

        for tile_idx in range(start_tile, end_tile + 1):
            tile_start = tile_idx * self.tile_size
            tile_end = min(tile_start + self.tile_size, offset + limit)

            # Create tile with dimensional context
            tile = TileKey(
                row_start=tile_start,
                row_end=tile_end,
                col_start=0,  # For now, full column width
                col_end=-1,   # -1 means all columns
                dimension_level={dim: i for i, dim in enumerate(row_dims)} if row_dims else None,
                drill_path=None
            )
            tiles.append(tile)

    # Check cache for each tile
    tiles_needed = []
    tiles_cached = []

    for tile in tiles:
        cache_key = self._cache_key_for_tile(tile, spec_dict)

        if self.cache.get(cache_key) is not None:
            tiles_cached.append(tile)
        else:
            tiles_needed.append(tile)

    strategy["can_reuse_tiles"] = len(tiles_cached) > 0
    strategy["tiles_needed"] = [t.to_string() for t in tiles_needed]
    strategy["cache_hits"] = len(tiles_cached)
    strategy["total_tiles"] = len(tiles)

    # Advanced: Implement tile prefetching for better performance
    if tiles_cached and self.enable_tiles:
        # Prefetch adjacent tiles that might be needed soon
        prefetch_strategy = _calculate_prefetch_tiles(tiles_cached, spec_dict, self.tile_size)
        if prefetch_strategy:
            strategy["prefetch_tiles"] = prefetch_strategy

    if not tiles_needed:
        # All tiles cached - return empty query list
        return [], strategy

    # Generate queries for missing tiles with optimizations
    tile_queries = []
    for tile in tiles_needed:
        # Only process aggregate queries for tiling
        for q in queries:
            if q.get("purpose") == "aggregate":
                # Create tile-specific query
                tile_q = self._create_tile_query(q, tile, spec_dict)
                tile_q["tile"] = tile  # Add tile information for tracking

                # Add query optimization for tiles
                tile_q = _optimize_tile_query(tile_q, tile)

                tile_queries.append(tile_q)

    return tile_queries, strategy


def _calculate_prefetch_tiles(
    self,
    cached_tiles: List[TileKey],
    spec_dict: Dict[str, Any]
) -> List[str]:
    """
    Calculate which tiles to prefetch based on cached tiles and access patterns.
    """
    prefetch_tiles = []
    tile_size = self.tile_size  # Use the instance's tile_size

    # For each cached tile, consider prefetching adjacent tiles
    for tile in cached_tiles:
        # Prefetch next tile (the one that would come after this in sequence)
        next_tile = TileKey(
            row_start=tile.row_end,
            row_end=tile.row_end + tile_size,
            col_start=tile.col_start,
            col_end=tile.col_end,
            dimension_level=tile.dimension_level,
            drill_path=tile.drill_path
        )
        prefetch_tiles.append(next_tile.to_string())

        # Prefetch previous tile (if not at start)
        if tile.row_start > 0:
            prev_start = max(0, tile.row_start - tile_size)
            prev_tile = TileKey(
                row_start=prev_start,
                row_end=tile.row_start,
                col_start=tile.col_start,
                col_end=tile.col_end,
                dimension_level=tile.dimension_level,
                drill_path=tile.drill_path
            )
            prefetch_tiles.append(prev_tile.to_string())

    return list(set(prefetch_tiles))  # Remove duplicates


    async def _compute_true_total(self, spec: PivotSpec, plan: Dict[str, Any]) -> int:
        """
        Compute the true total row count using a COUNT query.
        This addresses the TODO in the pagination result.
        """
        # Build a count query based on the spec
        # This would execute a SELECT COUNT(*) query to get the true total
        # without the LIMIT applied in the original query

        # For the count query, we need to match the same filters and conditions
        # as the original query but count instead of returning data
        table_name = spec.table
        where_clause = ""
        params = []

        if spec.filters:
            from ..util.sql_builder import build_where_clause
            where_clause, params = build_where_clause(spec.filters)
            where_clause = f"WHERE {where_clause}"

        count_sql = f"SELECT COUNT(*) as total_rows FROM {table_name} {where_clause}"

        try:
            count_query = {"sql": count_sql, "params": params}
            count_result = await self.backend.execute(count_query)

            if count_result and count_result.num_rows > 0:
                import pyarrow.compute as pc
                total_val = count_result.column(0)[0]
                if hasattr(total_val, 'as_py'):
                    return total_val.as_py()
                else:
                    return int(total_val)
            else:
                return 0
        except:
            # If count query fails, return the current count
            return 0

def _optimize_tile_query(
    query: Dict[str, Any],
    tile: TileKey
) -> Dict[str, Any]:
    """
    Apply query-level optimizations specific to tile access.
    """
    # Basic optimization - ensure query is properly limited to tile boundaries
    # In a real implementation, this would include more sophisticated optimizations
    return query


# Update the tile-aware method to use the enhanced multi-dimensional planning
QueryDiffEngine._plan_multi_dimensional = _plan_multi_dimensional
QueryDiffEngine._calculate_prefetch_tiles = _calculate_prefetch_tiles
QueryDiffEngine._optimize_tile_query = _optimize_tile_query
# Update the original method to use the new enhanced version
QueryDiffEngine._plan_tile_aware = _plan_multi_dimensional