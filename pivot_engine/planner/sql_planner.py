"""
SQLPlanner - Optimized for TanStack Query/Table pivot operations with Advanced Planning.

Key improvements:
- Support for HAVING clauses (post-aggregation filters)
- Multiple sort fields
- Column totals and subtotals
- Percentile and statistical aggregations
- Window functions for rankings
- Query optimization hints
- Null handling options
- Time-based grouping (date truncation)
- Expression-based measures
- Cross-tabulation support
- Advanced Planning: Cost-based optimization, plan selection, query rewriting
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re

# Import types from the main types module to avoid duplication
from ..types.pivot_spec import PivotSpec, Measure, NullHandling

# ==================== Advanced Planning Components ====================

@dataclass
class QueryPlan:
    """Represents a single query execution plan"""
    sql: str
    params: List[Any]
    purpose: str
    cost_estimate: float
    execution_steps: List[str]
    optimization_applied: List[str]


@dataclass
class PlanMetadata:
    """Metadata about the planning process"""
    original_plan: Dict[str, Any]
    alternative_plans: List[QueryPlan]
    selected_plan: QueryPlan
    optimization_strategy: str
    statistics: Dict[str, Any]


class CostEstimator:
    """Estimates query execution cost based on various factors"""

    @staticmethod
    def estimate_base_cost(num_rows: int, num_filters: int, num_grouping_cols: int,
                          num_measures: int, has_joins: bool = False) -> float:
        """Base cost estimation formula"""
        # Base cost based on data size
        data_cost = num_rows * 0.01  # Base cost per row

        # Cost multiplier based on complexity
        complexity_multiplier = (
            1.0 +  # Base
            num_filters * 0.15 +  # Filter cost
            num_grouping_cols * 0.25 +  # Grouping cost
            num_measures * 0.20  # Aggregation cost
        )

        # Additional cost for joins
        if has_joins:
            complexity_multiplier *= 2.0

        return data_cost * complexity_multiplier

    @staticmethod
    def estimate_with_table_stats(
        table_name: str,
        filters: List[Dict[str, Any]],
        grouping_cols: List[str],
        measures: List[Measure],
        table_stats: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate cost using table statistics if available"""
        if table_stats and 'row_count' in table_stats:
            num_rows = table_stats['row_count']
        else:
            # Default estimate - could be made configurable
            num_rows = 100000

        return CostEstimator.estimate_base_cost(
            num_rows, len(filters), len(grouping_cols), len(measures)
        )


class QueryRewriter:
    """Applies optimization rules to rewrite queries for better performance"""

    @staticmethod
    def rewrite_for_performance(
        sql: str,
        spec: PivotSpec,  # Using imported PivotSpec
        backend_type: str = "duckdb"
    ) -> Tuple[str, List[str]]:
        """Apply performance optimization rules to the SQL query"""
        optimizations_applied = []

        # Rule 1: Push down filters where possible
        optimized_sql = QueryRewriter._push_down_filters(sql, spec, backend_type)
        if optimized_sql != sql:
            optimizations_applied.append("push_down_filters")
            sql = optimized_sql

        # Rule 2: Optimize WHERE clauses (simplify redundant conditions)
        optimized_sql = QueryRewriter._simplify_where_clause(sql, spec, backend_type)
        if optimized_sql != sql:
            optimizations_applied.append("simplify_where_clause")
            sql = optimized_sql

        # Rule 3: Index hint optimization (if supported by backend)
        optimized_sql = QueryRewriter._add_index_hints(sql, spec, backend_type)
        if optimized_sql != sql:
            optimizations_applied.append("add_index_hints")
            sql = optimized_sql

        # Rule 4: Optimize aggregation order if possible
        optimized_sql = QueryRewriter._optimize_aggregation_order(sql, spec, backend_type)
        if optimized_sql != sql:
            optimizations_applied.append("optimize_aggregation_order")
            sql = optimized_sql

        return sql, optimizations_applied

    @staticmethod
    def _push_down_filters(sql: str, spec: PivotSpec, backend_type: str) -> str:
        """Push filters as close to the data source as possible"""
        # In the current implementation, filters are already well-positioned
        # This is a placeholder for more complex pushdown logic
        return sql

    @staticmethod
    def _simplify_where_clause(sql: str, spec: PivotSpec, backend_type: str) -> str:
        """Simplify WHERE clauses by removing redundant conditions"""
        # Placeholder for more sophisticated WHERE clause optimization
        return sql

    @staticmethod
    def _add_index_hints(sql: str, spec: PivotSpec, backend_type: str) -> str:
        """Add index hints if the backend supports them"""
        # This would be more relevant for systems like PostgreSQL, MySQL
        # For now, return the original SQL
        return sql

    @staticmethod
    def _optimize_aggregation_order(sql: str, spec: PivotSpec, backend_type: str) -> str:
        """Optimize aggregation order for performance"""
        # Placeholder for aggregation order optimization
        return sql

# Supported filter operators
_SIMPLE_OPS = {"=": "=", "==": "=", "!=": "<>", "<>": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<="}
_SET_OPS = {"in": "IN", "not in": "NOT IN"}
_OTHER_OPS = {"between": "BETWEEN", "like": "LIKE", "ilike": "ILIKE", "is null": "IS NULL", "is not null": "IS NOT NULL"}
_PATTERN_OPS = {"starts_with": "LIKE", "ends_with": "LIKE", "contains": "LIKE"}

class AggregationType(Enum):
    """Supported aggregation types"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    STDDEV = "stddev"
    VARIANCE = "variance"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    FIRST = "first"
    LAST = "last"
    STRING_AGG = "string_agg"
    ARRAY_AGG = "array_agg"

def safe_ident(name: str) -> str:
    """Validate and quote SQL identifiers to prevent injection."""
    if not name or not isinstance(name, str):
        raise ValueError(f"Invalid identifier: {name}")
    # Remove dangerous characters
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        # DuckDB uses double quotes for identifiers
        escaped = name.replace('"', '""')
        return f'"{escaped}"'
    return name

def validate_alias(alias: str) -> str:
    """Validate that an alias is safe (used for measure aliases, etc.)"""
    if not alias or not isinstance(alias, str):
        raise ValueError(f"Invalid alias: {alias}")
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', alias):
        raise ValueError(f"Alias must be alphanumeric with underscores: {alias}")
    return alias
    
class SQLPlanner:
    """
    Advanced SQL query planner for pivot table operations.
    Optimized for DuckDB with support for complex aggregations.
    Enhanced with advanced planning capabilities: cost estimation and query optimization.
    """

    def __init__(self, dialect: str = "duckdb", enable_optimization: bool = True):
        self.dialect = dialect
        self._query_cache = {}
        self.enable_optimization = enable_optimization
        self.cost_estimator = CostEstimator()
        self.query_rewriter = QueryRewriter()
    
    def plan(
        self,
        spec: PivotSpec,
        *,
        columns_top_n: Optional[int] = None,
        columns_order_by_measure: Optional[Measure] = None,
        include_metadata: bool = True,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an optimized query plan for the given PivotSpec with advanced planning capabilities.

        Returns:
        {
            "queries": [
                {"name": str, "sql": str, "params": list, "purpose": str, "estimated_cost": float},
                ...
            ],
            "metadata": {
                "group_by": list,
                "agg_aliases": list,
                "column_expr": str or None,
                "has_window_functions": bool,
                "estimated_complexity": str,
                "optimization_enabled": bool,
                "total_estimated_cost": float
            }
        }
        """
        queries = []
        metadata = {}

        # Validate spec
        self._validate_spec(spec)
        table_ident = safe_ident(spec.table)

        # Build WHERE clause (pre-aggregation filters)
        where_sql, where_params = self._build_where(spec.filters)

        # Build measures
        agg_sql_list, agg_aliases, has_window_funcs = self._build_measures(spec.measures)

        # Build group-by
        group_cols = list(spec.rows) + list(spec.columns)
        group_by_sql = self._build_group_by(group_cols)

        # Store metadata
        metadata["group_by"] = group_cols
        metadata["agg_aliases"] = agg_aliases
        metadata["has_window_functions"] = has_window_funcs

        # Handle column dimension top-N query
        column_expr = None
        if spec.columns:
            column_expr = self._build_column_expr(spec.columns)
            metadata["column_expr"] = column_expr

            if columns_top_n and columns_top_n > 0:
                col_query = self._build_column_values_query(
                    table_ident, column_expr, where_sql, where_params,
                    columns_top_n, columns_order_by_measure
                )
                queries.append(col_query)

        # Build main aggregation query
        main_query = self._build_main_query(
            spec, table_ident, group_by_sql, agg_sql_list,
            where_sql, where_params, agg_aliases
        )
        queries.append(main_query)

        # Build totals query if requested
        if spec.totals:
            totals_query = self._build_totals_query(
                spec, table_ident, agg_sql_list, where_sql, where_params
            )
            if totals_query:
                queries.append(totals_query)

        # Estimate query complexity
        if include_metadata:
            metadata["estimated_complexity"] = self._estimate_complexity(
                len(group_cols), len(spec.measures), len(spec.filters),
                has_window_funcs, spec.totals
            )

        # Apply advanced planning optimization if enabled
        plan_result = {"queries": queries, "metadata": metadata}
        if self.enable_optimization and optimize:
            plan_result = self._apply_advanced_planning(plan_result, spec)
        else:
            # Add basic optimization metadata even when disabled
            if "metadata" not in plan_result:
                plan_result["metadata"] = {}
            plan_result["metadata"]["optimization_enabled"] = False
            plan_result["metadata"]["advanced_planning_applied"] = False
            # Still add estimated_cost to queries if not already present
            for query in plan_result.get("queries", []):
                if "estimated_cost" not in query:
                    query["estimated_cost"] = 0.0  # Default cost when not calculated

        return plan_result

    def _apply_advanced_planning(self, plan_result: Dict[str, Any], spec: PivotSpec) -> Dict[str, Any]:
        """
        Apply advanced planning optimizations to the plan result.
        """
        # Calculate cost estimates for each query
        for query in plan_result.get("queries", []):
            query["estimated_cost"] = self._estimate_query_cost(query, spec)

        # Apply query rewriting for performance optimization
        for query in plan_result.get("queries", []):
            if "sql" in query:
                original_sql = query["sql"]
                optimized_sql, optimizations = self.query_rewriter.rewrite_for_performance(
                    original_sql, spec, self.dialect
                )
                if optimized_sql != original_sql:
                    query["sql"] = optimized_sql
                    query["optimization_applied"] = optimizations
                    # Re-estimate cost after optimization
                    query["estimated_cost"] = self._estimate_query_cost(query, spec)

        # Add optimization metadata
        total_estimated_cost = sum(
            q.get("estimated_cost", 0) for q in plan_result.get("queries", [])
        )

        if "metadata" not in plan_result:
            plan_result["metadata"] = {}
        plan_result["metadata"]["optimization_enabled"] = True
        plan_result["metadata"]["total_estimated_cost"] = total_estimated_cost
        plan_result["metadata"]["advanced_planning_applied"] = True

        return plan_result

    def _estimate_query_cost(self, query: Dict[str, Any], spec: PivotSpec) -> float:
        """Estimate the execution cost of a single query"""
        filters = spec.filters
        grouping_cols = spec.rows + spec.columns
        measures = spec.measures  # Use all measures for cost calculation

        has_joins = "JOIN" in (query.get("sql", "") or "").upper()

        # Use table statistics or default estimates
        return self.cost_estimator.estimate_base_cost(
            100000,  # Default row estimate
            len(filters),
            len(grouping_cols),
            len(measures),
            has_joins
        )
    
    def _validate_spec(self, spec: PivotSpec):
        """Validate the PivotSpec structure"""
        if not spec.table:
            raise ValueError("PivotSpec must include a 'table' name")
        if not spec.measures:
            raise ValueError("PivotSpec must include at least one measure")
        
        # Validate measure aliases are unique
        aliases = [m.alias for m in spec.measures]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Measure aliases must be unique")
    
    def _build_where(self, filters: List[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """Build parameterized WHERE clause with enhanced operator support"""
        if not filters:
            return "", []
        
        clauses = []
        params: List[Any] = []
        
        for f in filters:
            field = f.get("field")
            op = (f.get("op") or "=").lower()
            val = f.get("value")
            
            if not field:
                continue
            
            ident = safe_ident(field)
            
            # Simple comparison operators
            if op in _SIMPLE_OPS:
                clauses.append(f"{ident} {_SIMPLE_OPS[op]} ?")
                params.append(val)
            
            # Set operators (IN, NOT IN)
            elif op in _SET_OPS:
                if not isinstance(val, (list, tuple, set)):
                    raise ValueError(f"Filter op '{op}' requires a list/tuple value")
                if not val:
                    # Empty IN clause - handle edge case
                    clauses.append("1=0" if op == "in" else "1=1")
                else:
                    placeholders = ", ".join(["?"] * len(val))
                    clauses.append(f"{ident} {_SET_OPS[op]} ({placeholders})")
                    params.extend(list(val))
            
            # NULL checks
            elif op in {"is null", "is not null"}:
                clauses.append(f"{ident} {_OTHER_OPS[op]}")
            
            # BETWEEN
            elif op == "between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError("BETWEEN filter requires two-element list")
                clauses.append(f"{ident} BETWEEN ? AND ?")
                params.extend([val[0], val[1]])
            
            # Pattern matching
            elif op in {"like", "ilike"}:
                clauses.append(f"{ident} {_OTHER_OPS[op]} ?")
                params.append(val)
            
            elif op in _PATTERN_OPS:
                # Convenience operators
                if op == "starts_with":
                    pattern = f"{val}%"
                elif op == "ends_with":
                    pattern = f"%{val}"
                else:  # contains
                    pattern = f"%{val}%"
                clauses.append(f"{ident} LIKE ?")
                params.append(pattern)
            
            else:
                raise NotImplementedError(f"Unsupported filter operator: {op}")
        
        where_sql = " AND ".join(clauses) if clauses else ""
        return where_sql, params
    
    def _build_having(self, having: List[Dict[str, Any]], agg_aliases: List[str]) -> Tuple[str, List[Any]]:
        """Build HAVING clause for post-aggregation filters"""
        if not having:
            return "", []
        
        # Similar to WHERE but validates against aggregate aliases
        clauses = []
        params = []
        
        for h in having:
            field = h.get("field")
            if field not in agg_aliases:
                raise ValueError(f"HAVING field '{field}' must be an aggregate alias")
            
            op = (h.get("op") or "=").lower()
            val = h.get("value")
            ident = safe_ident(field)
            
            if op in _SIMPLE_OPS:
                clauses.append(f"{ident} {_SIMPLE_OPS[op]} ?")
                params.append(val)
            elif op == "between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError("BETWEEN requires two-element list")
                clauses.append(f"{ident} BETWEEN ? AND ?")
                params.extend([val[0], val[1]])
            else:
                raise NotImplementedError(f"HAVING operator '{op}' not supported")
        
        having_sql = " AND ".join(clauses) if clauses else ""
        return having_sql, params
    
    def _build_measures(self, measures: List[Measure]) -> Tuple[List[str], List[str], bool]:
        """
        Build aggregate SQL fragments.
        Returns (sql_list, aliases, has_window_functions)
        """
        if not measures:
            return ["COUNT(*) AS count__"], ["count__"], False
        
        agg_sql_list = []
        aliases = []
        has_window_funcs = False
        
        for m in measures:
            sql, has_window = self._measure_to_sql(m)
            agg_sql_list.append(sql)
            aliases.append(m.alias)
            if has_window:
                has_window_funcs = True
        
        return agg_sql_list, aliases, has_window_funcs
    
    def _measure_to_sql(self, m: Measure) -> Tuple[str, bool]:
        """
        Convert Measure to SQL fragment.
        Returns (sql, has_window_function)
        """
        has_window = False
        
        # Use custom expression if provided
        if m.expression:
            sql = f"{m.expression} AS {safe_ident(m.alias)}"
            has_window = "OVER" in m.expression.upper()
            return sql, has_window
        
        field_ident = safe_ident(m.field) if m.field else "*"
        agg = m.agg.strip().lower()
        alias_ident = safe_ident(m.alias)
        
        # Apply null handling wrapper if needed
        if m.null_handling == NullHandling.AS_ZERO and agg not in {"count", "count_distinct"}:
            field_expr = f"COALESCE({field_ident}, 0)"
        elif m.null_handling == NullHandling.AS_EMPTY and agg in {"string_agg", "array_agg"}:
            field_expr = f"COALESCE({field_ident}, '')"
        else:
            field_expr = field_ident
        
        # Build aggregation based on type
        if agg in {"sum", "avg", "min", "max"}:
            agg_func = agg.upper()
            sql = f"{agg_func}({field_expr}) AS {alias_ident}"
        
        elif agg == "count":
            if m.field in {None, "*"}:
                sql = f"COUNT(*) AS {alias_ident}"
            else:
                sql = f"COUNT({field_expr}) AS {alias_ident}"
        
        elif agg in {"count_distinct", "distinct_count"}:
            sql = f"COUNT(DISTINCT {field_expr}) AS {alias_ident}"
        
        elif agg == "stddev":
            sql = f"STDDEV({field_expr}) AS {alias_ident}"
        
        elif agg == "variance":
            sql = f"VARIANCE({field_expr}) AS {alias_ident}"
        
        elif agg == "median":
            sql = f"MEDIAN({field_expr}) AS {alias_ident}"
        
        elif agg == "percentile":
            if m.percentile is None:
                raise ValueError("Percentile aggregation requires percentile parameter")
            sql = f"QUANTILE({field_expr}, {m.percentile}) AS {alias_ident}"
        
        elif agg == "string_agg":
            sep = m.separator or ','
            sql = f"STRING_AGG({field_expr}, '{sep}') AS {alias_ident}"
        
        elif agg == "array_agg":
            sql = f"ARRAY_AGG({field_expr}) AS {alias_ident}"
        
        elif agg in {"first", "last"}:
            # Use window functions for FIRST/LAST
            sql = f"{agg.upper()}({field_expr}) AS {alias_ident}"
        
        else:
            # Custom function - validate and use as-is
            func = safe_ident(agg)
            sql = f"{func}({field_expr}) AS {alias_ident}"
        
        # Add FILTER clause if specified
        if m.filter_condition:
            sql = sql.replace(f" AS {alias_ident}", f" FILTER (WHERE {m.filter_condition}) AS {alias_ident}")
        
        return sql, has_window
    
    def _build_group_by(self, group_cols: List[str]) -> str:
        """Build GROUP BY clause"""
        if not group_cols:
            return ""
        return ", ".join([safe_ident(c) for c in group_cols])
    
    def _build_column_expr(self, columns: List[str]) -> str:
        """Create composite key expression for column dimensions"""
        if not columns:
            raise ValueError("columns must be non-empty")
        
        if len(columns) == 1:
            return f"CAST({safe_ident(columns[0])} AS VARCHAR)"
        
        parts = [f"CAST({safe_ident(col)} AS VARCHAR)" for col in columns]
        return " || '|' || ".join(parts)
    
    def _build_column_values_query(
        self, table_ident: str, column_expr: str,
        where_sql: str, where_params: List[Any],
        top_n: int, order_measure: Optional[Measure]
    ) -> Dict[str, Any]:
        """Build query to fetch top-N column values"""
        if order_measure:
            measure_sql, _ = self._measure_to_sql(order_measure)
            sql = (
                f"SELECT {column_expr} AS _col_key, {measure_sql} "
                f"FROM {table_ident}"
            )
            if where_sql:
                sql += f" WHERE {where_sql}"
            sql += f" GROUP BY {column_expr} ORDER BY {safe_ident(order_measure.alias)} DESC LIMIT {top_n}"
        else:
            sql = f"SELECT DISTINCT {column_expr} AS _col_key FROM {table_ident}"
            if where_sql:
                sql += f" WHERE {where_sql}"
            sql += f" LIMIT {top_n}"
        
        return {
            "name": "column_values",
            "sql": sql,
            "params": where_params,
            "purpose": "column_values"
        }
    
    def _build_main_query(
        self, spec: PivotSpec, table_ident: str,
        group_by_sql: str, agg_sql_list: List[str],
        where_sql: str, where_params: List[Any],
        agg_aliases: List[str]
    ) -> Dict[str, Any]:
        """Build the main aggregation query"""
        # SELECT clause
        agg_sql = ", ".join(agg_sql_list)
        select_cols = f"{group_by_sql}, " if group_by_sql else ""
        sql = f"SELECT {select_cols}{agg_sql} FROM {table_ident}"
        
        # WHERE clause
        params = []
        if where_sql:
            sql += f" WHERE {where_sql}"
            params.extend(where_params)
        
        # GROUP BY clause
        if group_by_sql:
            sql += f" GROUP BY {group_by_sql}"
        
        # HAVING clause (post-aggregation filters)
        if spec.having:
            having_sql, having_params = self._build_having(spec.having, agg_aliases)
            if having_sql:
                sql += f" HAVING {having_sql}"
                params.extend(having_params)
        
        # ORDER BY clause
        order_sql, order_params = self._build_order(spec.sort, agg_aliases)
        if order_sql:
            sql += f" ORDER BY {order_sql}"
            params.extend(order_params)
        
        # LIMIT/OFFSET (pagination)
        if hasattr(spec, 'limit') and spec.limit:
            limit = min(int(spec.limit), 1000000)
            sql += f" LIMIT {limit}"
        
        return {
            "name": "aggregate",
            "sql": sql,
            "params": params,
            "purpose": "aggregate"
        }
    
    def _build_order(
        self, sort: Optional[Union[Dict, List[Dict]]], agg_aliases: List[str]
    ) -> Tuple[str, List[Any]]:
        """Build ORDER BY clause with multi-field support"""
        if not sort:
            return "", []
        
        # Normalize to list
        sort_list = [sort] if isinstance(sort, dict) else sort
        
        order_clauses = []
        for s in sort_list:
            field = s.get("field")
            if not field:
                continue
            
            order = (s.get("order") or "asc").upper()
            if order not in {"ASC", "DESC"}:
                order = "ASC"
            
            # Handle nulls
            nulls = s.get("nulls", "").upper()
            nulls_clause = ""
            if nulls in {"FIRST", "LAST"}:
                nulls_clause = f" NULLS {nulls}"
            
            order_clauses.append(f"{safe_ident(field)} {order}{nulls_clause}")
        
        return ", ".join(order_clauses) if order_clauses else "", []
    
    def _build_totals_query(
        self, spec: PivotSpec, table_ident: str,
        agg_sql_list: List[str], where_sql: str, where_params: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """Build query for grand totals"""
        if not agg_sql_list:
            return None
        
        agg_sql = ", ".join(agg_sql_list)
        sql = f"SELECT {agg_sql} FROM {table_ident}"
        
        if where_sql:
            sql += f" WHERE {where_sql}"
        
        return {
            "name": "totals",
            "sql": sql,
            "params": where_params,
            "purpose": "totals"
        }
    
    def _estimate_complexity(
        self, num_groups: int, num_measures: int,
        num_filters: int, has_windows: bool, has_totals: bool
    ) -> str:
        """Estimate query complexity for optimization hints"""
        score = num_groups * 2 + num_measures + num_filters
        if has_windows:
            score += 10
        if has_totals:
            score += 5
        
        if score < 10:
            return "low"
        elif score < 25:
            return "medium"
        else:
            return "high"