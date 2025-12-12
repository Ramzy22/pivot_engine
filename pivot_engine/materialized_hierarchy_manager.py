"""
MaterializedHierarchyManager - Pre-compute and store hierarchical rollups for common drill paths
"""
import asyncio
from typing import Dict, Any, List, Optional
import ibis
from pivot_engine.types.pivot_spec import PivotSpec


class MaterializedHierarchyManager:
    def __init__(self, backend, cache):
        self.backend = backend # Expects an Ibis connection
        self.cache = cache
        self.rollup_tables = {}
        
    def create_materialized_hierarchy(self, spec: PivotSpec):
        """Create materialized hierarchy for common drill paths using Ibis."""
        hierarchy_name = f"hierarchy_{spec.table}_{abs(hash(str(spec.to_dict()))):x}"
        base_table = self.backend.table(spec.table)

        for level in range(1, len(spec.rows) + 1):
            level_dims = spec.rows[:level]
            rollup_table_name = f"{hierarchy_name}_level_{level}"

            # Define aggregations in Ibis
            aggregations = []
            for m in spec.measures:
                agg_func = getattr(base_table[m.field], m.agg)
                aggregations.append(agg_func().name(m.alias))

            # Build the Ibis expression for the rollup
            rollup_expr = base_table.group_by(level_dims).aggregate(aggregations)

            # Create the table in the database
            self.backend.create_table(rollup_table_name, rollup_expr, overwrite=True)
            
            self.rollup_tables[f"{spec.table}:{level}"] = rollup_table_name
    
    def get_rollup_table_name(self, spec: PivotSpec, level: int) -> Optional[str]:
        """Get the name of the rollup table for a given level."""
        return self.rollup_tables.get(f"{spec.table}:{level}")


class IntelligentPrefetchManager:
    def __init__(self, session_tracker, pattern_analyzer, backend, cache):
        self.session_tracker = session_tracker
        self.pattern_analyzer = pattern_analyzer
        self.backend = backend
        self.cache = cache
        self.prefetch_cache = {}
        
    async def determine_prefetch_strategy(self, user_session: Dict[str, Any], current_spec: PivotSpec, expanded_paths: List[List[str]]):
        """Determine intelligent prefetching based on user behavior patterns"""
        # If pattern analyzer is not available, use simple heuristics
        user_patterns = {}
        if self.pattern_analyzer:
            try:
                user_patterns = await self.pattern_analyzer.analyze_session(
                    user_session, current_spec
                )
            except:
                # Use default patterns if analyzer fails
                user_patterns = {'frequently_expanded': [], 'common_drill_paths': []}
        else:
            # Use default patterns
            user_patterns = {'frequently_expanded': [], 'common_drill_paths': []}

        # Determine likely next expansions
        likely_paths = await self._predict_likely_expansions(
            user_patterns, current_spec, expanded_paths
        )

        # Pre-fetch likely paths
        prefetch_tasks = []
        for path in likely_paths:
            task = self._prefetch_path_data(current_spec, path)
            prefetch_tasks.append(task)

        # Execute prefetch tasks in background
        await asyncio.gather(*prefetch_tasks, return_exceptions=True)

        return likely_paths
    
    async def _predict_likely_expansions(self, patterns: Dict[str, Any], spec: PivotSpec, current_paths: List[List[str]]):
        """Predict which paths user is likely to expand next based on behavioral patterns"""
        predictions = []
        
        # Example prediction logic based on common drill patterns
        for path in current_paths:
            # If this path has been expanded before, user likely to expand children
            path_key = str(path)
            if path_key in patterns.get('frequently_expanded', []):
                next_level = len(path) + 1
                if next_level < len(spec.rows):
                    # Add wildcard prediction for all children of this path
                    predictions.append(path + ["*"])  # Wildcard for all children
        
        # Additional predictions based on data patterns
        # For example, if certain dimensions are commonly expanded together
        common_drill_paths = patterns.get('common_drill_paths', [])
        for common_path in common_drill_paths:
            if current_paths and common_path.startswith(tuple(current_paths[0]) if current_paths else ()):
                predictions.append(common_path)
        
        return predictions[:5]  # Limit to top 5 predictions
    
    async def _prefetch_path_data(self, spec: PivotSpec, path: List[str]):
        """Pre-fetch data for a specific path"""
        if len(path) >= len(spec.rows):
            return  # Can't prefetch beyond the hierarchy depth
        
        # Build query to get next level data
        next_level_index = len(path)
        if next_level_index < len(spec.rows):
            next_dimension = spec.rows[next_level_index]
            
            # Build parent filters
            filter_conditions = []
            filter_params = []
            for i, val in enumerate(path):
                if i < len(spec.rows):
                    filter_conditions.append(f"{spec.rows[i]} = ?")
                    filter_params.append(val)
            
            where_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ""
            
            query = f"""
            SELECT {next_dimension},
                   {', '.join([f"{m.agg}({m.field}) as {m.alias}" for m in spec.measures])}
            FROM {spec.table}
            {where_clause}
            GROUP BY {next_dimension}
            ORDER BY {next_dimension}
            LIMIT 100
            """
            
            try:
                result = await self.backend.execute({'sql': query, 'params': filter_params})
                # Cache the prefetched data
                cache_key = f"prefetch:{hash(str(spec.to_dict()))}:{str(path)}"
                await self.cache.set(cache_key, result, ttl=300)
                self.prefetch_cache[cache_key] = result
            except Exception as e:
                print(f"Prefetch failed for path {path}: {str(e)}")
    
    def get_cached_prefetch(self, spec: PivotSpec, path: List[str]):
        """Get cached prefetched data"""
        cache_key = f"prefetch:{hash(str(spec.to_dict()))}:{str(path)}"
        return self.prefetch_cache.get(cache_key)
    
    async def warm_prefetch_cache(self, spec: PivotSpec, user_session: Dict[str, Any]):
        """Warm the prefetch cache based on user patterns"""
        patterns = await self.pattern_analyzer.analyze_session(user_session, spec)
        likely_paths = await self._predict_likely_expansions(patterns, spec, [])
        
        prefetch_tasks = [self._prefetch_path_data(spec, path) for path in likely_paths]
        await asyncio.gather(*prefetch_tasks, return_exceptions=True)