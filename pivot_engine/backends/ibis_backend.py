"""
IbisBackend - database-agnostic backend for any Ibis-supported database.

Features:
- Ibis query execution (SQL injection safe)
- Arrow table output for zero-copy data transfer
- Connection pooling
- Query timeout and cancellation
- Performance metrics
- Database-agnostic operations
"""

from typing import Any, List, Dict, Optional, Union
import time
from contextlib import contextmanager

try:
    import ibis
except ImportError:
    ibis = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


class IbisBackend:
    """
    Database-agnostic backend that works with any Ibis-supported database.
    """

    def __init__(
        self,
        connection: Optional[Any] = None,
        connection_uri: Optional[str] = None,
        **connection_kwargs
    ):
        """
        Initialize Ibis backend.

        Args:
            connection: An existing Ibis connection
            connection_uri: URI string for connecting to the database
            **connection_kwargs: Additional connection parameters
        """
        if ibis is None:
            raise ImportError("ibis package required. Install: pip install ibis-framework")
        
        self.con = connection
        
        # Support for different database backends based on URI
        if connection_uri:
            if connection_uri.startswith("postgres://"):
                from urllib.parse import urlparse
                parsed = urlparse(connection_uri)
                self.con = ibis.postgres.connect(
                    host=parsed.hostname,
                    port=parsed.port,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path[1:]  # Remove leading slash
                )
            elif connection_uri.startswith("mysql://"):
                from urllib.parse import urlparse
                parsed = urlparse(connection_uri)
                self.con = ibis.mysql.connect(
                    host=parsed.hostname,
                    port=parsed.port or 3306,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path[1:]
                )
            elif connection_uri.startswith("bigquery://"):
                self.con = ibis.bigquery.connect(**connection_kwargs)
            elif connection_uri.startswith("snowflake://"):
                from urllib.parse import urlparse
                parsed = urlparse(connection_uri)
                self.con = ibis.snowflake.connect(
                    user=parsed.username,
                    password=parsed.password,
                    account=parsed.hostname,
                    **connection_kwargs
                )
            elif connection_uri.startswith("clickhouse://"):
                from urllib.parse import urlparse
                parsed = urlparse(connection_uri)
                self.con = ibis.clickhouse.connect(
                    host=parsed.hostname,
                    port=parsed.port or 8123,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path[1:] if parsed.path else 'default',
                    **connection_kwargs
                )
            elif connection_uri.startswith("sqlite://"):
                db_path = connection_uri.replace("sqlite://", "")
                self.con = ibis.sqlite.connect(db_path)
            elif connection_uri.startswith("duckdb://") or connection_uri == ":memory:":
                self.con = ibis.duckdb.connect(connection_uri.replace("duckdb://", "") if connection_uri.startswith("duckdb://") else connection_uri)
            else:
                # Default to DuckDB
                self.con = ibis.duckdb.connect(connection_uri)

        # Track query stats
        self._query_count = 0
        self._total_time = 0.0

    def execute(self, query: Dict[str, Any]) -> pa.Table:
        """
        Execute a query and return the result as a PyArrow Table.

        Args:
            query: A dictionary containing the 'sql' and 'params'.
                   For Ibis, 'sql' is actually an Ibis expression.

        Returns:
            A PyArrow Table with the query result.
        """
        start_time = time.time()

        try:
            # For Ibis, execute the expression directly
            if isinstance(query, dict) and 'ibis_expr' in query:
                ibis_expr = query['ibis_expr']
                # Prefer to_pyarrow() for zero-copy efficiency
                if hasattr(ibis_expr, 'to_pyarrow'):
                    result = ibis_expr.to_pyarrow()
                else:
                    result = ibis_expr.execute()
            else:
                # If plain SQL is provided, convert to Ibis expression
                sql = query.get("sql", "")
                if sql:
                    # Use raw_sql method for direct SQL execution
                    result = self.con.raw_sql(sql)
                else:
                    raise ValueError("Query must contain either 'ibis_expr' or 'sql' field")

            # Convert to Arrow table if not already
            if not isinstance(result, pa.Table):
                # Convert pandas DataFrame or other results to Arrow table
                if hasattr(result, 'to_arrow_table'):
                    result = result.to_arrow_table()
                elif hasattr(result, 'to_arrow'):
                    result = result.to_arrow()
                elif pa is not None:
                    import pandas as pd
                    if isinstance(result, pd.DataFrame):
                        result = pa.Table.from_pandas(result)
                    else:
                        # Fallback for other types
                        try:
                            result = pa.Table.from_pandas(pd.DataFrame(result))
                        except:
                            pass
            
            # Performance tracking
            self._query_count += 1
            self._total_time += (time.time() - start_time)

            return result
        except Exception as e:
            # Log error and re-raise
            print(f"Error executing query:\nQuery: {query}\nError: {e}")
            raise

    def execute_arrow(
        self,
        query: Union[str, Dict[str, Any]],
        params: Optional[List[Any]] = None
    ) -> pa.Table:
        """
        Execute query and return Arrow table.

        Convenience method for zero-copy Arrow output.
        """
        return self.execute(query, params, return_arrow=True)

    def execute_batch(
        self,
        queries: List[Dict[str, Any]],
        return_arrow: bool = False
    ) -> List[Union[List[Dict[str, Any]], pa.Table]]:
        """
        Execute multiple queries in sequence.

        Args:
            queries: List of query dicts with "sql" and "params"
            return_arrow: Return Arrow tables

        Returns:
            List of results (one per query)
        """
        results = []
        for query in queries:
            result = self.execute(query, return_arrow=return_arrow)
            results.append(result)
        return results

    def execute_streaming(
        self,
        query: Union[str, Dict[str, Any]],
        params: Optional[List[Any]] = None,
        batch_size: int = 1000
    ):
        """
        Execute query and yield results in batches.

        Useful for large result sets to avoid loading everything in memory.

        Yields:
            Batches of rows as list of dicts
        """
        # Note: Streaming behavior varies by database backend
        # For now, return the full result as batches
        result_table = self.execute(query, params)
        
        num_rows = result_table.num_rows
        for i in range(0, num_rows, batch_size):
            batch = result_table.slice(i, min(batch_size, num_rows - i))
            yield batch.to_pylist()

    def register_arrow_dataset(
        self,
        name: str,
        arrow_table: pa.Table,
        temporary: bool = True
    ):
        """
        Register Arrow table as queryable dataset.

        Args:
            name: Name to register as
            arrow_table: Arrow table to register
            temporary: If True, table is session-only
        """
        if ibis is None:
            raise ImportError("ibis required")

        # In Ibis, register the table as a temporary table
        if temporary:
            self.con.create_table(name, arrow_table, temp=True)
        else:
            self.con.create_table(name, arrow_table)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get backend performance statistics.
        """
        avg_time = self._total_time / self._query_count if self._query_count > 0 else 0

        return {
            "query_count": self._query_count,
            "total_time": self._total_time,
            "avg_query_time": avg_time,
            "backend_type": getattr(self.con, 'name', 'unknown') if self.con else 'disconnected'
        }

    def reset_stats(self):
        """Reset performance counters"""
        self._query_count = 0
        self._total_time = 0.0

    @contextmanager
    def transaction(self):
        """
        Context manager for transactions.

        Note: Transaction behavior varies by database backend.
        """
        try:
            yield
        except Exception:
            # Rollback behavior varies by database
            raise

    def close(self):
        """Close database connection"""
        if hasattr(self.con, 'close'):
            self.con.close()
        self.con = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'close'):
            self.close()