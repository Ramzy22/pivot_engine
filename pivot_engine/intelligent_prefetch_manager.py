"""
IntelligentPrefetchManager and related components for optimizing data fetching.
"""
from typing import Dict, Any, List, Optional
import asyncio
import ibis
from ibis.expr.api import Table as IbisTable
from pivot_engine.types.pivot_spec import PivotSpec

class UserPatternAnalyzer:
    def __init__(self, cache):
        self.cache = cache

    async def analyze_patterns(self, user_session: Dict[str, Any]) -> List[List[str]]:
        """
        Analyzes user behavior patterns to predict future data access.
        Simple heuristic: Returns paths that have been drilled into frequently (> 2 times).
        """
        history = user_session.get('history', [])
        if not history:
            return []
            
        # Count frequency of paths in history
        # History is expected to be a list of lists of strings (paths)
        path_counts = {}
        for path in history:
            # Convert list path to tuple for hashing
            path_tuple = tuple(path)
            path_counts[path_tuple] = path_counts.get(path_tuple, 0) + 1
            
        # Return paths with frequency > 2
        frequent_paths = [list(p) for p, count in path_counts.items() if count > 2]
        return frequent_paths

class IntelligentPrefetchManager:
    def __init__(self, session_tracker: Any, pattern_analyzer: UserPatternAnalyzer, backend: Any, cache: Any):
        self.session_tracker = session_tracker
        self.pattern_analyzer = pattern_analyzer
        self.backend = backend # Ibis connection
        self.cache = cache

    async def determine_prefetch_strategy(self, user_session: Dict[str, Any], spec: PivotSpec, expanded_paths: List[List[str]]) -> List[List[str]]:
        """
        Determines which data paths to prefetch based on user patterns and current pivot state.
        Strategically combines:
        1. Paths predicted by user history analysis.
        2. Immediate children of currently expanded paths (next logical step).
        """
        # 1. Get predicted paths from analyzer
        predicted_paths = await self.pattern_analyzer.analyze_patterns(user_session)
        
        # 2. Add immediate children of currently expanded paths
        # Logic: If a user has expanded 'Region: North', they likely want to see the cities in North.
        # We need to find what the next dimension is for each expanded path.
        
        paths_to_fetch = []
        
        # Combine predicted and current expanded paths
        candidates = expanded_paths + predicted_paths
        
        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for c in candidates:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                unique_candidates.append(c)
        
        for path in unique_candidates:
            # Determine the depth of this path
            depth = len(path)
            
            # Check if there is a next level in the rows hierarchy
            if depth < len(spec.rows):
                next_dim = spec.rows[depth] # The dimension to fetch values for
                
                # We want to fetch the top N values for this next dimension, 
                # effectively pre-loading the children nodes.
                try:
                    children = await self._fetch_top_children(spec.table, spec.rows[:depth], path, next_dim)
                    for child_val in children:
                        new_path = path + [child_val]
                        paths_to_fetch.append(new_path)
                except Exception as e:
                    print(f"Error prefetching children for path {path}: {e}")
                    
        return paths_to_fetch

    async def _fetch_top_children(self, table_name: str, parent_dims: List[str], parent_values: List[str], target_dim: str, limit: int = 5) -> List[Any]:
        """
        Queries the database to find the top values for the next dimension.
        """
        # Using Ibis to construct the query
        try:
            t = self.backend.table(table_name)
            
            # Filter by parent path
            query = t
            for dim, val in zip(parent_dims, parent_values):
                query = query.filter(t[dim] == val)
            
            # Select distinct target dimension values
            # Optimization: In a real scenario, we might order by a measure (e.g., top sales)
            # For now, just distinct values
            query = query.select(target_dim).distinct().limit(limit)
            
            # Execute
            # Note: This is synchronous in standard Ibis, but wrapped in async loop in controller if needed.
            # Here we assume we can call it. If backend supports async, use it.
            if hasattr(self.backend, 'execute'):
                 # Standard Ibis
                 result = query.execute()
                 return result[target_dim].tolist()
            else:
                 # Fallback
                 return []
        except Exception as e:
            # print(f"Prefetch query failed: {e}")
            return []
