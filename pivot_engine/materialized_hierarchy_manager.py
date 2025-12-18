"""
MaterializedHierarchyManager - Pre-compute and store hierarchical rollups for common drill paths
"""
import asyncio
from typing import Dict, Any, List, Optional
import ibis
from ibis import BaseBackend as IbisBaseBackend
from pivot_engine.types.pivot_spec import PivotSpec


class MaterializedHierarchyManager:
    def __init__(self, backend: IbisBaseBackend, cache):
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