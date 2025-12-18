"""
query_planning_service.py - Microservice for query planning in distributed architecture
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from pivot_engine.planner.sql_planner import SQLPlanner
from pivot_engine.planner.ibis_planner import IbisPlanner
from pivot_engine.types.pivot_spec import PivotSpec, Measure, NullHandling
from pivot_engine.backends.duckdb_backend import DuckDBBackend

try:
    import ibis
except ImportError:
    ibis = None

class QueryPlanningService:
    """Service for query planning in distributed architecture"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize backend connection if URI provided
        self.con = None
        if self.config.get('backend_uri'):
            try:
                # Use simple DuckDB backend for planning if not specified
                self.con = ibis.duckdb.connect(self.config['backend_uri'])
            except Exception as e:
                print(f"Warning: Could not connect to backend for planning: {e}")

        self.planners = {
            'sql': SQLPlanner(dialect=self.config.get('sql_dialect', 'duckdb')),
            'ibis': IbisPlanner(con=self.con)
        }
        self.cost_estimator = CostEstimator()
        
        # Simple catalog for materialized views (simulated)
        self.mv_catalog = {}
        
    async def plan_pivot_query(self, spec_dict: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """Plan pivot query using appropriate planner"""
        spec = PivotSpec.from_dict(spec_dict)
        
        # Determine best planner based on spec characteristics
        planner_key = await self._select_optimal_planner(spec, context)
        planner = self.planners.get(planner_key, self.planners['sql'])
        
        try:
            plan = planner.plan(spec)
            
            # If using IbisPlanner, plan['queries'] are Ibis expressions
            # If using SQLPlanner, plan['queries'] are dicts with SQL
            
            # Apply cost-based optimization
            optimized_plan = await self._apply_optimization_plan(plan, spec, context)
            
            # Helper for tests/consumers expecting just the expression/query
            # If it's a single query plan, return the query object directly if requested
            # (Matches test expectation of returning ibis.Expr)
            if context and context.get('return_expr', False):
                 if optimized_plan.get('queries'):
                     return optimized_plan['queries'][0]

            return optimized_plan
            
        except Exception as e:
            # Fallback to SQL planner if Ibis fails (e.g. no connection)
            if planner_key == 'ibis':
                print(f"Ibis planning failed ({e}), falling back to SQL")
                return self.planners['sql'].plan(spec)
            raise e

    async def _select_optimal_planner(self, spec: PivotSpec, context: Optional[Dict[str, Any]] = None) -> str:
        """Select optimal planner based on data characteristics"""
        # Prefer Ibis if connection is available
        if self.con is not None:
            return 'ibis'
            
        # Get table size estimate if available in context
        table_size = (context or {}).get('table_size', 100000)
        
        # Simple heuristic
        return 'sql'

    async def _apply_optimization_plan(self, plan: Dict[str, Any], spec: PivotSpec, context: Optional[Dict[str, Any]] = None):
        """Apply cost-based optimization to the plan"""
        # Calculate initial cost
        initial_cost = await self.cost_estimator.estimate_cost(plan, spec, context)
        
        # Apply optimization strategies
        if spec.rows and len(spec.rows) > 2:
            # For deep hierarchies, consider materialized views
            plan = await self._apply_materialized_view_optimization(plan, spec)
        
        # Recalculate cost after optimization
        final_cost = await self.cost_estimator.estimate_cost(plan, spec, context)
        
        if 'metadata' not in plan:
            plan['metadata'] = {}
            
        plan['metadata']['optimization_applied'] = True
        plan['metadata']['initial_cost_estimate'] = initial_cost
        plan['metadata']['final_cost_estimate'] = final_cost
        plan['metadata']['cost_reduction'] = initial_cost - final_cost
        
        return plan
    
    async def _apply_materialized_view_optimization(self, plan: Dict[str, Any], spec: PivotSpec):
        """Apply materialized view optimization for hierarchies"""
        # Check if materialized views exist for this spec
        if await self._has_materialized_view_for_spec(spec):
            mv_name = await self._get_materialized_view_name(spec)
            
            queries = plan.get('queries', [])
            new_queries = []
            
            for query in queries:
                # Logic to swap table depends on query type (SQL dict vs Ibis expr)
                if hasattr(query, 'mutate'): # Ibis expression
                     # Replacing table in Ibis expression is complex without rebuilding
                     # For now, we assume if MV exists, we just return a plan that queries it directly
                     # This is a simplification
                     pass
                elif isinstance(query, dict) and 'sql' in query:
                    # Replace table name in SQL
                    query['sql'] = query['sql'].replace(spec.table, mv_name)
                    
                new_queries.append(query)
            
            plan['queries'] = new_queries
            if 'metadata' not in plan: plan['metadata'] = {}
            plan['metadata']['used_materialized_view'] = mv_name
        
        return plan
    
    async def _has_materialized_view_for_spec(self, spec: PivotSpec) -> bool:
        """Check if materialized view exists for spec"""
        mv_name = self._generate_mv_name(spec)
        return mv_name in self.mv_catalog
    
    async def _get_materialized_view_name(self, spec: PivotSpec) -> str:
        """Get materialized view name for spec"""
        return self._generate_mv_name(spec)
        
    def _generate_mv_name(self, spec: PivotSpec) -> str:
        return f"mv_{spec.table}_{abs(hash(str(spec.rows)))}"

    def register_materialized_view(self, spec: PivotSpec, name: str):
        """Register a materialized view"""
        mv_name = self._generate_mv_name(spec)
        self.mv_catalog[mv_name] = name


class CostEstimator:
    """Cost estimator for query planning"""
    
    async def estimate_cost(self, plan: Dict[str, Any], spec: PivotSpec, context: Optional[Dict[str, Any]] = None) -> float:
        """Estimate execution cost of a plan"""
        context = context or {}
        
        # Base cost based on data size
        table_size = context.get('table_size', 100000)
        base_cost = table_size * 0.001  # Base cost per row
        
        # Cost multipliers based on complexity
        complexity_multiplier = 1.0
        
        # Number of grouping dimensions
        complexity_multiplier += len(spec.rows) * 0.1
        
        # Number of measures
        complexity_multiplier += len(spec.measures) * 0.05
        
        # Number of filters
        complexity_multiplier += len(spec.filters) * 0.02
        
        # Number of sort fields
        sort_fields = spec.sort if isinstance(spec.sort, list) else [spec.sort] if spec.sort else []
        complexity_multiplier += len(sort_fields) * 0.03
        
        # Check for expensive operations
        for measure in spec.measures:
            if measure.agg in ['median', 'percentile']:
                complexity_multiplier += 0.5  # Expensive operations
        
        if spec.columns:
            complexity_multiplier += 0.2  # Pivot operations are more expensive
        
        total_cost = base_cost * complexity_multiplier
        return total_cost
