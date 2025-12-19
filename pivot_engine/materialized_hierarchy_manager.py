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
        self.jobs = {}  # job_id -> {status, progress, error, result}

    async def create_materialized_hierarchy_async(self, spec: PivotSpec) -> str:
        """
        Start an asynchronous job to create materialized hierarchy.
        Returns a job_id.
        """
        import uuid
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "pending", "progress": 0, "table": spec.table}
        
        # Run in a separate thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, self._create_materialized_hierarchy_sync, spec, job_id)
        
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a materialization job."""
        return self.jobs.get(job_id, {"status": "unknown"})

    def _create_materialized_hierarchy_sync(self, spec: PivotSpec, job_id: Optional[str] = None):
        """Internal synchronous worker to create materialized hierarchy."""
        try:
            if job_id:
                self.jobs[job_id]["status"] = "running"
            
            hierarchy_name = f"hierarchy_{spec.table}_{abs(hash(str(spec.to_dict()))):x}"
            base_table = self.backend.table(spec.table)
            
            total_levels = len(spec.rows)
            
            for level in range(1, total_levels + 1):
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
                # Note: 'overwrite=True' might not work on all backends, but Ibis handles it generally
                self.backend.create_table(rollup_table_name, rollup_expr, overwrite=True)
                
                # OPTIMIZATION: Create Index on the grouping columns (dimensions)
                # This is crucial for performance with millions of rows
                # We try to access the raw connection or use a helper if available
                self._create_index_safely(rollup_table_name, level_dims)
                
                self.rollup_tables[f"{spec.table}:{level}"] = rollup_table_name
                
                if job_id:
                    self.jobs[job_id]["progress"] = int((level / total_levels) * 100)

            if job_id:
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["hierarchy_name"] = hierarchy_name
                
        except Exception as e:
            print(f"Materialization failed: {e}")
            if job_id:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(e)
                
    def _create_index_safely(self, table_name: str, columns: List[str]):
        """Helper to create indexes if the backend supports it"""
        # Check if the backend object passed to __init__ has the 'create_index' method
        # (which we added to IbisBackend wrapper, but self.backend here is the raw Ibis connection object usually)
        # Wait, in ScalablePivotController, we pass 'con' which is the raw Ibis connection.
        # But we added 'create_index' to 'IbisBackend' wrapper class.
        # We need to bridge this gap. 
        
        # Approach 1: Try to execute raw SQL on the connection object directly
        try:
            import hashlib
            cols_str = "_".join(columns)
            hash_suffix = hashlib.md5(cols_str.encode()).hexdigest()[:8]
            index_name = f"idx_{table_name}_{hash_suffix}"[:63]
            
            safe_table = table_name
            safe_cols = ", ".join([f'"{c}"' for c in columns])
            sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{safe_table}" ({safe_cols})'
            
            if hasattr(self.backend, 'raw_sql'):
                 self.backend.raw_sql(sql)
            elif hasattr(self.backend, 'execute'):
                 self.backend.execute(sql)
        except Exception as e:
            # Index creation is optional/optimization, so we log and continue
            print(f"Index creation skipped for {table_name}: {e}")

    def create_materialized_hierarchy(self, spec: PivotSpec):
        """Legacy synchronous method (keeps backward compatibility)."""
        self._create_materialized_hierarchy_sync(spec)
    
    def get_rollup_table_name(self, spec: PivotSpec, level: int) -> Optional[str]:
        """Get the name of the rollup table for a given level."""
        return self.rollup_tables.get(f"{spec.table}:{level}")