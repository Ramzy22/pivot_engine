"""
execution_service.py - Distributed query execution service
"""
import asyncio
from typing import Dict, Any, List, Optional
import pyarrow as pa
from ...backends.duckdb_backend import DuckDBBackend
from ...types.pivot_spec import PivotSpec

try:
    import ibis
except ImportError:
    ibis = None

class ExecutionService:
    """Service for distributed query execution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backends = []
        self.scheduler = QueryScheduler()
        self.partition_strategies = {
            'range': RangePartitionStrategy(),
            'hash': HashPartitionStrategy(),
            'dimension': DimensionPartitionStrategy()
        }
    
    async def execute_plan(self, plan: Any, spec: PivotSpec) -> pa.Table:
        """
        Execute a query plan.
        
        Args:
            plan: Query plan (dict or Ibis expression)
            spec: Pivot specification
        """
        # If plan is just an Ibis expression, wrap it
        if hasattr(plan, 'schema') or hasattr(plan, 'execute'):
             plan = {'queries': [plan]}
             
        return await self.execute_distributed_query(plan, spec)

    async def execute_distributed_query(self, plan: Dict[str, Any], spec: PivotSpec):
        """Execute query across multiple backend instances"""
        # Determine if query can be distributed
        if await self._can_distribute_query(plan, spec):
            # Split query into distributable parts
            sub_queries = await self._split_plan_for_distribution(plan, spec)
            
            # Execute in parallel across available backends
            results = await asyncio.gather(*[
                self._execute_on_available_backend(query) 
                for query in sub_queries
            ])
            
            # Merge results
            final_result = await self._merge_results(results, spec)
            return final_result
            
        else:
            # Execute on single backend
            return await self._execute_single_query(plan, spec)
    
    async def _can_distribute_query(self, plan: Dict[str, Any], spec: PivotSpec) -> bool:
        """Determine if a query can be distributed"""
        # For now, assume most aggregation queries can be distributed if we have partitions
        # and the plan contains queries
        queries = plan.get('queries', [])
        return len(queries) > 0 and len(spec.rows) > 0
    
    async def _split_plan_for_distribution(self, plan: Dict[str, Any], spec: PivotSpec) -> List[Any]:
        """Split plan into distributable parts"""
        queries = plan.get('queries', [])
        if not queries:
            return []
            
        partition_strategy = self.config.get('partition_strategy', 'dimension')
        
        if partition_strategy not in self.partition_strategies:
            partition_strategy = 'dimension'
        
        partitioner = self.partition_strategies[partition_strategy]
        partitions = await partitioner.create_partitions(spec)
        
        if not partitions:
            return queries # No partitions, return original
            
        sub_queries = []
        base_query = queries[0]
        
        for partition in partitions:
            partitioned_query = await self._apply_partition_to_query(
                base_query, partition, spec
            )
            sub_queries.append(partitioned_query)
        
        return sub_queries
    
    async def _apply_partition_to_query(self, query: Any, partition: Dict[str, Any], spec: PivotSpec) -> Any:
        """Apply partition to a query"""
        # Handle Ibis expression
        if hasattr(query, 'filter'):
            filters_list = partition.get('filters_list', [])
            if filters_list:
                # Apply filters to Ibis expression
                filtered_query = query
                for f in filters_list:
                    field = f.get('field')
                    op = f.get('op')
                    val = f.get('value')
                    
                    if hasattr(filtered_query, 'columns') and field in filtered_query.columns:
                        col = filtered_query[field]
                        if op == '>=':
                            filtered_query = filtered_query.filter(col >= val)
                        elif op == '<':
                            filtered_query = filtered_query.filter(col < val)
                        elif op == 'in':
                            filtered_query = filtered_query.filter(col.isin(val))
                        elif op == 'is_null':
                             filtered_query = filtered_query.filter(col.isnull())
                        elif op == 'hash_mod':
                             # This is backend specific, simplified for now
                             pass
                return filtered_query
            return query

        # Handle SQL dict (legacy)
        if isinstance(query, dict) and 'sql' in query:
            partition_filter = partition.get('filter', '')
            partition_params = partition.get('params', [])
            
            if partition_filter:
                sql = query['sql']
                if 'WHERE' in sql.upper():
                    pos = sql.upper().find('WHERE') + 5
                    modified_sql = sql[:pos] + f" AND ({partition_filter}) " + sql[pos:]
                else:
                    where_pos = sql.upper().find('GROUP BY')
                    if where_pos == -1: where_pos = sql.upper().find('ORDER BY')
                    if where_pos == -1: where_pos = len(sql)
                    
                    modified_sql = sql[:where_pos] + f" WHERE {partition_filter} " + sql[where_pos:]
                
                modified_params = query.get('params', []) + partition_params
                
                return {
                    'sql': modified_sql,
                    'params': modified_params,
                    'purpose': query.get('purpose', 'aggregate'),
                    'partition_info': partition
                }
        
        return query
    
    async def _execute_on_available_backend(self, query: Any) -> pa.Table:
        """Execute query on an available backend"""
        if not self.backends:
            # Create a default backend if none available
            backend = DuckDBBackend()
            self.backends.append(backend)
        
        backend = self.backends[0]
        
        # Handle Ibis expression execution via backend
        if hasattr(query, 'execute'):
             # If backend supports generic execute or we just run it directly
             # DuckDBBackend has .execute() that expects dict.
             # But if it's Ibis, we might want to convert to SQL or use Ibis execution.
             
             # If backend is DuckDBBackend, it expects dict with SQL. 
             # But Ibis expressions can be executed directly if we have a connection.
             # Since we don't have the Ibis connection here easily (it's inside IbisBackend or DuckDBBackend's underlying con)
             # We rely on the backend wrapper to handle it.
             
             # If using our IbisBackend:
             if hasattr(backend, 'execute') and hasattr(query, 'compile'):
                 # Create a wrapper dict
                 return backend.execute({'ibis_expr': query})
             elif hasattr(query, 'to_pyarrow'):
                 return query.to_pyarrow()
             else:
                 return query.execute()

        return await backend.execute(query)
    
    async def _execute_single_query(self, plan: Dict[str, Any], spec: PivotSpec) -> pa.Table:
        """Execute query on single backend"""
        queries = plan.get('queries', [])
        if not queries:
            return pa.table({})
        
        return await self._execute_on_available_backend(queries[0])
    
    async def _merge_results(self, results: List[pa.Table], spec: PivotSpec) -> pa.Table:
        """Merge results from distributed execution"""
        if not results:
            return pa.table({})
        
        if len(results) == 1:
            return results[0]
        
        try:
            # Check if all tables have the same schema
            first_schema = results[0].schema
            # Filter out empty tables
            valid_results = [t for t in results if t.num_rows > 0]
            if not valid_results:
                 return results[0]

            return pa.concat_tables(valid_results)
                    
        except Exception as e:
            print(f"Error merging results: {e}")
            return results[0]


class QueryScheduler:
    """Scheduler for query execution"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_workers = 0
        self.max_workers = 4
    
    async def schedule_query(self, query: Any, backend) -> pa.Table:
        """Schedule query execution"""
        future = asyncio.Future()
        await self.queue.put((query, backend, future))
        
        if self.active_workers < self.max_workers:
            asyncio.create_task(self._worker())
            self.active_workers += 1
        
        return await future
    
    async def _worker(self):
        """Worker to execute queries"""
        try:
            while True:
                query, backend, future = await self.queue.get()
                try:
                    result = await backend.execute(query)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            pass


class PartitionStrategy:
    """Base class for partition strategies"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        """Create partition definitions"""
        return [] # Default to no partitioning


class RangePartitionStrategy(PartitionStrategy):
    """Partition by range (e.g., time, numeric values)"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        # For this example, create 4 partitions
        # We use a structured format 'filters_list' for Ibis compatibility
        return [
            {'filter': 'id >= 0 AND id < 25000', 'filters_list': [{'field': 'id', 'op': '>=', 'value': 0}, {'field': 'id', 'op': '<', 'value': 25000}], 'id': 'p1'},
            {'filter': 'id >= 25000 AND id < 50000', 'filters_list': [{'field': 'id', 'op': '>=', 'value': 25000}, {'field': 'id', 'op': '<', 'value': 50000}], 'id': 'p2'},
            {'filter': 'id >= 50000 AND id < 75000', 'filters_list': [{'field': 'id', 'op': '>=', 'value': 50000}, {'field': 'id', 'op': '<', 'value': 75000}], 'id': 'p3'},
            {'filter': 'id >= 75000', 'filters_list': [{'field': 'id', 'op': '>=', 'value': 75000}], 'id': 'p4'}
        ]


class HashPartitionStrategy(PartitionStrategy):
    """Partition by hash of key values"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        # For this example, use hash of first dimension
        partition_field = spec.rows[0] if spec.rows else 'id'
        # Simplified filters for Ibis (not fully implementing hash mod in Ibis here)
        return [
            {'filter': f"HASH({partition_field}) % 4 = 0", 'filters_list': [{'field': partition_field, 'op': 'hash_mod', 'value': 0}], 'id': 'p1'},
             # ... simplified
        ]


class DimensionPartitionStrategy(PartitionStrategy):
    """Partition by dimension values"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        # For this example, partition by the first dimension values
        partition_field = spec.rows[0] if spec.rows else 'region'
        return [
            {'filter': f"{partition_field} IN ('North', 'South')", 'filters_list': [{'field': partition_field, 'op': 'in', 'value': ['North', 'South']}], 'id': 'p1'},
            {'filter': f"{partition_field} IN ('East', 'West')", 'filters_list': [{'field': partition_field, 'op': 'in', 'value': ['East', 'West']}], 'id': 'p2'},
            {'filter': f"{partition_field} IS NULL", 'filters_list': [{'field': partition_field, 'op': 'is_null', 'value': None}], 'id': 'p4'}
        ]
