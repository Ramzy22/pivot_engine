"""
execution_service.py - Distributed query execution service
"""
import asyncio
from typing import Dict, Any, List, Optional
import pyarrow as pa
from ...backends.duckdb_backend import DuckDBBackend
from ...types.pivot_spec import PivotSpec


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
        # Check if table supports partitioning
        # Check if query operations are distributable
        # Check if backends are available
        
        # For now, assume most aggregation queries can be distributed
        return len(plan.get('queries', [])) > 0 and len(spec.rows) > 0
    
    async def _split_plan_for_distribution(self, plan: Dict[str, Any], spec: PivotSpec) -> List[Dict[str, Any]]:
        """Split plan into distributable parts"""
        queries = plan.get('queries', [])
        partition_strategy = self.config.get('partition_strategy', 'range')
        
        if partition_strategy not in self.partition_strategies:
            partition_strategy = 'range'
        
        partitioner = self.partition_strategies[partition_strategy]
        partitions = await partitioner.create_partitions(spec)
        
        sub_queries = []
        for partition in partitions:
            partitioned_query = await self._apply_partition_to_query(
                queries[0] if queries else {}, partition, spec
            )
            sub_queries.append(partitioned_query)
        
        return sub_queries
    
    async def _apply_partition_to_query(self, query: Dict[str, Any], partition: Dict[str, Any], spec: PivotSpec) -> Dict[str, Any]:
        """Apply partition to a query"""
        # Add partition-based filters to the query
        partition_filter = partition.get('filter', '')
        partition_params = partition.get('params', [])
        
        if partition_filter:
            # Insert partition filter into the WHERE clause
            sql = query['sql']
            if 'WHERE' in sql.upper():
                # Insert after WHERE
                pos = sql.upper().find('WHERE') + 5  # After 'WHERE'
                modified_sql = sql[:pos] + f" AND ({partition_filter}) " + sql[pos:]
            else:
                # Add WHERE clause
                where_pos = sql.upper().find('GROUP BY')
                if where_pos == -1:
                    where_pos = sql.upper().find('ORDER BY')
                if where_pos == -1:
                    where_pos = len(sql)
                
                modified_sql = sql[:where_pos] + f" WHERE {partition_filter} " + sql[where_pos:]
            
            # Add partition parameters to query parameters
            modified_params = query.get('params', []) + partition_params
            
            return {
                'sql': modified_sql,
                'params': modified_params,
                'purpose': query.get('purpose', 'aggregate'),
                'partition_info': partition
            }
        
        return query
    
    async def _execute_on_available_backend(self, query: Dict[str, Any]) -> pa.Table:
        """Execute query on an available backend"""
        # For this implementation, use the first available backend
        # In a real implementation, this would use connection pooling and load balancing
        if not self.backends:
            # Create a default backend if none available
            backend = DuckDBBackend()
            self.backends.append(backend)
        
        backend = self.backends[0]  # Use first backend for this example
        return await backend.execute(query)
    
    async def _execute_single_query(self, plan: Dict[str, Any], spec: PivotSpec) -> pa.Table:
        """Execute query on single backend"""
        queries = plan.get('queries', [])
        if not queries:
            return pa.table({})
        
        # Use the first query to execute
        return await self._execute_on_available_backend(queries[0])
    
    async def _merge_results(self, results: List[pa.Table], spec: PivotSpec) -> pa.Table:
        """Merge results from distributed execution"""
        if not results:
            return pa.table({})
        
        if len(results) == 1:
            return results[0]
        
        # For aggregation queries, we need to merge the aggregations appropriately
        # This is complex and depends on the aggregation types
        # For this example, we'll just concatenate (which works for group-by queries)
        
        try:
            # Check if all tables have the same schema
            first_schema = results[0].schema
            all_same_schema = all(table.schema.equals(first_schema) for table in results)
            
            if all_same_schema:
                # Concatenate tables if they have the same schema
                return pa.concat_tables(results)
            else:
                # If schemas differ, we need to merge at the application level
                # This is a simplified approach
                merged_data = {}
                
                # Collect all data
                for table in results:
                    for i in range(table.num_rows):
                        row = {}
                        for col_name in table.column_names:
                            val = table.column(col_name)[i].as_py()
                            row[col_name] = val
                        # Use the first few dimensions as the key
                        key_parts = [str(row.get(dim, '')) for dim in spec.rows[:3] if dim in row]
                        key = '|'.join(key_parts)
                        merged_data[key] = row
                
                # Convert back to table format
                if merged_data:
                    all_rows = list(merged_data.values())
                    if all_rows:
                        # Create a table from the merged data
                        columns = {}
                        for col in all_rows[0].keys():
                            columns[col] = [row.get(col) for row in all_rows]
                        
                        return pa.table({k: pa.array(v) for k, v in columns.items()})
                else:
                    return results[0]
                    
        except Exception as e:
            print(f"Error merging results: {e}")
            # If merge fails, return first result as fallback
            return results[0]


class QueryScheduler:
    """Scheduler for query execution"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_workers = 0
        self.max_workers = 4
    
    async def schedule_query(self, query: Dict[str, Any], backend) -> pa.Table:
        """Schedule query execution"""
        future = asyncio.Future()
        await self.queue.put((query, backend, future))
        
        # Start worker if needed
        if self.active_workers < self.max_workers:
            asyncio.create_task(self._worker())
            self.active_workers += 1
        
        return await future
    
    async def _worker(self):
        """Worker to execute queries"""
        try:
            query, backend, future = await self.queue.get()
            try:
                result = await backend.execute(query)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        finally:
            self.queue.task_done()


class PartitionStrategy:
    """Base class for partition strategies"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        """Create partition definitions"""
        raise NotImplementedError


class RangePartitionStrategy(PartitionStrategy):
    """Partition by range (e.g., time, numeric values)"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        # For this example, create 4 partitions
        return [
            {'filter': 'id >= 0 AND id < 25000', 'params': [], 'id': 'p1'},
            {'filter': 'id >= 25000 AND id < 50000', 'params': [], 'id': 'p2'},
            {'filter': 'id >= 50000 AND id < 75000', 'params': [], 'id': 'p3'},
            {'filter': 'id >= 75000', 'params': [], 'id': 'p4'}
        ]


class HashPartitionStrategy(PartitionStrategy):
    """Partition by hash of key values"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        # For this example, use hash of first dimension
        partition_field = spec.rows[0] if spec.rows else 'id'
        return [
            {'filter': f"HASH({partition_field}) % 4 = 0", 'params': [], 'id': 'p1'},
            {'filter': f"HASH({partition_field}) % 4 = 1", 'params': [], 'id': 'p2'},
            {'filter': f"HASH({partition_field}) % 4 = 2", 'params': [], 'id': 'p3'},
            {'filter': f"HASH({partition_field}) % 4 = 3", 'params': [], 'id': 'p4'}
        ]


class DimensionPartitionStrategy(PartitionStrategy):
    """Partition by dimension values"""
    
    async def create_partitions(self, spec: PivotSpec) -> List[Dict[str, Any]]:
        # For this example, partition by the first dimension values
        # In a real implementation, this would query the database for distinct values
        partition_field = spec.rows[0] if spec.rows else 'region'
        return [
            {'filter': f"{partition_field} IN ('North', 'South')", 'params': [], 'id': 'p1'},
            {'filter': f"{partition_field} IN ('East', 'West')", 'params': [], 'id': 'p2'},
            {'filter': f"{partition_field} IN ('Central', 'Other')", 'params': [], 'id': 'p3'},
            {'filter': f"{partition_field} IS NULL", 'params': [], 'id': 'p4'}
        ]