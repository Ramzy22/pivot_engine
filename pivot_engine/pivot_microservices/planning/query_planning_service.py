"""
query_planning_service.py - Microservice for query planning in distributed architecture
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from ....planner.sql_planner import SQLPlanner
from ....planner.ibis_planner import IbisPlanner
from ....types.pivot_spec import PivotSpec, Measure, NullHandling


class QueryPlanningService:
    """Service for query planning in distributed architecture"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.planners = {
            'sql': SQLPlanner(dialect=config.get('sql_dialect', 'duckdb')),
            'ibis': IbisPlanner()  # Will be initialized with connection
        }
        self.cost_estimator = CostEstimator()
        
    async def plan_pivot_query(self, spec_dict: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """Plan pivot query using appropriate planner"""
        spec = PivotSpec.from_dict(spec_dict)
        
        # Determine best planner based on spec characteristics
        planner = await self._select_optimal_planner(spec, context)
        plan = await planner.plan(spec)
        
        # Apply cost-based optimization
        optimized_plan = await self._apply_optimization_plan(plan, spec, context)
        
        return optimized_plan

    async def _select_optimal_planner(self, spec: PivotSpec, context: Optional[Dict[str, Any]] = None):
        """Select optimal planner based on data characteristics"""
        # Get table size estimate if available in context
        table_size = (context or {}).get('table_size', 100000)  # Default estimate
        
        if table_size > 10000000:  # 10M+ rows
            # For very large tables, consider distributed planning
            return self.planners['sql']  # Could be enhanced with distributed planner
        elif len(spec.rows) > 3:  # Deep hierarchy
            return self.planners['sql']  # Optimized for hierarchies
        else:
            return self.planners['sql']

    async def _apply_optimization_plan(self, plan: Dict[str, Any], spec: PivotSpec, context: Optional[Dict[str, Any]] = None):
        """Apply cost-based optimization to the plan"""
        # Calculate initial cost
        initial_cost = await self.cost_estimator.estimate_cost(plan, spec, context)
        
        # Apply optimization strategies
        if spec.rows and len(spec.rows) > 2:
            # For deep hierarchies, consider materialized views
            plan = await self._apply_materialized_view_optimization(plan, spec)
        
        if spec.filters and len(spec.filters) > 3:
            # For many filters, optimize filter order
            plan = await self._apply_filter_optimization(plan, spec)
        
        # Recalculate cost after optimization
        final_cost = await self.cost_estimator.estimate_cost(plan, spec, context)
        
        plan['metadata']['optimization_applied'] = True
        plan['metadata']['initial_cost_estimate'] = initial_cost
        plan['metadata']['final_cost_estimate'] = final_cost
        plan['metadata']['cost_reduction'] = initial_cost - final_cost
        
        return plan
    
    async def _apply_materialized_view_optimization(self, plan: Dict[str, Any], spec: PivotSpec):
        """Apply materialized view optimization for hierarchies"""
        # Check if materialized views exist for this spec
        if await self._has_materialized_view_for_spec(spec):
            # Modify plan to use materialized view
            for query in plan.get('queries', []):
                if query.get('purpose') == 'aggregate':
                    # Replace table with materialized view
                    mv_name = await self._get_materialized_view_name(spec)
                    # This is simplified - real implementation would substitute table names
                    pass
        
        return plan
    
    async def _apply_filter_optimization(self, plan: Dict[str, Any], spec: PivotSpec):
        """Apply filter optimization"""
        # Reorder filters by selectivity to apply most selective first
        # This would involve analyzing filter cardinality and reordering
        return plan
    
    async def _has_materialized_view_for_spec(self, spec: PivotSpec) -> bool:
        """Check if materialized view exists for spec"""
        # Placeholder implementation
        return False
    
    async def _get_materialized_view_name(self, spec: PivotSpec) -> str:
        """Get materialized view name for spec"""
        return f"mv_{spec.table}_{hash(str(spec.to_dict()))}"


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