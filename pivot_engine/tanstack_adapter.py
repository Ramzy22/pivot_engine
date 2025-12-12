"""
tanstack_adapter.py - Direct TanStack Table/Query adapter for the scalable pivot engine
This bypasses the REST API and provides direct integration with TanStack components
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from pivot_engine.scalable_pivot_controller import ScalablePivotController
from pivot_engine.types.pivot_spec import PivotSpec, Measure


class TanStackOperation(str, Enum):
    """TanStack operation types"""
    GET_DATA = "get_data"
    GET_ROWS = "get_rows"
    GET_COLUMNS = "get_columns"
    GET_PAGE_COUNT = "get_page_count"
    FILTER = "filter"
    SORT = "sort"
    GROUP = "group"


@dataclass
class TanStackRequest:
    """TanStack request structure"""
    operation: TanStackOperation
    table: str
    columns: List[Dict[str, Any]]
    filters: List[Dict[str, Any]]
    sorting: List[Dict[str, Any]]
    grouping: List[str]
    aggregations: List[Dict[str, Any]]
    pagination: Optional[Dict[str, Any]] = None
    global_filter: Optional[str] = None


@dataclass
class TanStackResponse:
    """TanStack response structure"""
    data: List[Dict[str, Any]]
    columns: List[Dict[str, Any]]
    pagination: Optional[Dict[str, Any]] = None
    total_rows: Optional[int] = None
    grouping: Optional[List[Dict[str, Any]]] = None


class TanStackPivotAdapter:
    """Direct TanStack adapter that bypasses REST API and connects to controller"""
    
    def __init__(self, controller: ScalablePivotController):
        self.controller = controller
        self.hierarchy_state = {}  # Store expansion state
    
    def convert_tanstack_request_to_pivot_spec(self, request: TanStackRequest) -> PivotSpec:
        """Convert TanStack request to PivotSpec format"""
        # Extract grouping columns as hierarchy
        hierarchy_cols = request.grouping or []
        
        # Extract measure columns
        measures = []
        value_cols = []
        
        for col in request.columns:
            if col.get('aggregationFn'):
                # This is an aggregation column
                measures.append(Measure(
                    field=col.get('aggregationField', col['id']),
                    agg=col.get('aggregationFn', 'sum'),
                    alias=col['id']
                ))
            elif col['id'] not in hierarchy_cols:
                # This is a value column
                value_cols.append(col['id'])
        
        # Convert TanStack filters to PivotSpec filters
        pivot_filters = []
        for tanstack_filter in request.filters:
            # TanStack filter format: {id: str, value: any, type?: str}
            field = tanstack_filter['id']
            value = tanstack_filter['value']
            operator = self._map_tanstack_operator(tanstack_filter.get('type', 'eq'))
            
            pivot_filters.append({
                'field': field,
                'op': operator,
                'value': value
            })
        
        # Convert TanStack sorting to PivotSpec sorting
        pivot_sort = []
        for sort_spec in request.sorting:
            pivot_sort.append({
                'field': sort_spec['id'],
                'order': 'asc' if sort_spec.get('desc', False) is False else 'desc'
            })
        
        # Handle pagination
        offset = 0
        limit = 1000  # Default
        
        if request.pagination:
            page_size = request.pagination.get('pageSize', 100)
            page = request.pagination.get('pageIndex', 0)
            offset = page * page_size
            limit = page_size
        
        return PivotSpec(
            table=request.table,
            rows=hierarchy_cols,
            columns=[],  # TanStack handles column pivoting differently
            measures=measures,
            filters=pivot_filters,
            sort=pivot_sort,
            limit=limit,
            totals=True  # Enable totals computation
        )
    
    def _map_tanstack_operator(self, tanstack_op: str) -> str:
        """Map TanStack filter operators to pivot engine operators"""
        mapping = {
            'eq': '=',
            'ne': '!=',
            'lt': '<',
            'gt': '>',
            'lte': '<=',
            'gte': '>=',
            'contains': 'contains',
            'startsWith': 'starts_with',
            'endsWith': 'ends_with',
            'in': 'in',
            'notIn': 'not in'
        }
        return mapping.get(tanstack_op, '=')
    
    def convert_pivot_result_to_tanstack_format(self, pivot_result: Dict[str, Any], 
                                               tanstack_request: TanStackRequest) -> TanStackResponse:
        """Convert pivot engine result to TanStack format"""
        # Convert from pivot format to TanStack row format
        rows = []
        
        if isinstance(pivot_result, dict) and 'rows' in pivot_result and 'columns' in pivot_result:
            # Dict format with columns and rows
            pivot_columns = pivot_result['columns']
            pivot_rows = pivot_result['rows']
            
            for pivot_row in pivot_rows:
                tanstack_row = {}
                for i, col_name in enumerate(pivot_columns):
                    tanstack_row[col_name] = pivot_row[i] if i < len(pivot_row) else None
                rows.append(tanstack_row)
        elif isinstance(pivot_result, list):
            # Already in row format
            rows = pivot_result
        
        # Calculate pagination info if needed
        pagination = None
        if tanstack_request.pagination:
            total_rows = len(rows)  # In real implementation, get actual total
            pagination = {
                'totalRows': total_rows,
                'pageSize': tanstack_request.pagination.get('pageSize', 100),
                'pageIndex': tanstack_request.pagination.get('pageIndex', 0),
                'pageCount': (total_rows + tanstack_request.pagination.get('pageSize', 100) - 1) // tanstack_request.pagination.get('pageSize', 100) if total_rows else 0
            }
        
        return TanStackResponse(
            data=rows,
            columns=tanstack_request.columns,
            pagination=pagination,
            total_rows=len(rows) if rows else 0
        )
    
    async def handle_request(self, request: TanStackRequest) -> TanStackResponse:
        """Handle a TanStack request directly"""
        # Convert request to pivot spec
        pivot_spec = self.convert_tanstack_request_to_pivot_spec(request)
        
        # Execute pivot operation
        pivot_result = self.controller.run_pivot(pivot_spec, return_format="dict")
        
        # Convert result to TanStack format
        tanstack_result = self.convert_pivot_result_to_tanstack_format(pivot_result, request)
        
        return tanstack_result
    
    async def handle_hierarchical_request(self, request: TanStackRequest, 
                                        expanded_paths: List[List[str]],
                                        user_preferences: Optional[Dict[str, Any]] = None) -> TanStackResponse:
        """Handle hierarchical TanStack request with expansion state"""
        pivot_spec = self.convert_tanstack_request_to_pivot_spec(request)
        
        # Use controller's hierarchical method
        hierarchy_result = self.controller.run_hierarchical_pivot(pivot_spec.to_dict())
        
        # Apply expansion state
        filtered_data = self._apply_expansion_state(hierarchy_result, expanded_paths)
        
        # Convert to TanStack format
        tanstack_result = self.convert_pivot_result_to_tanstack_format(
            filtered_data, request
        )
        
        return tanstack_result
    
    def _apply_expansion_state(self, hierarchy_result: Dict[str, Any], 
                              expanded_paths: List[List[str]]) -> Dict[str, Any]:
        """Apply expansion state to hierarchy result"""
        # For now, return the result as-is; in a full implementation,
        # this would filter the result based on expansion state
        return hierarchy_result
    
    def get_schema_info(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for TanStack column configuration"""
        # This would query the database schema
        # For now, return mock schema based on the controller's data
        try:
            # Run a simple query to get column info
            spec = PivotSpec(
                table=table_name,
                rows=[],  # No grouping
                measures=[],
                filters=[]
            )
            sample_result = self.controller.run_pivot(spec, return_format="dict")
            
            if sample_result and sample_result.get('columns'):
                columns_info = []
                for col_name in sample_result['columns']:
                    # Determine column type (simplified)
                    col_type = "string"  # Default
                    # In a real implementation, this would analyze the data
                    columns_info.append({
                        'id': col_name,
                        'header': col_name.replace('_', ' ').title(),
                        'accessorKey': col_name,
                        'type': col_type,
                        'enableSorting': True,
                        'enableFiltering': True
                    })
                
                return {
                    'table': table_name,
                    'columns': columns_info,
                    'sample_data': sample_result.get('rows', [])[:5]  # First 5 rows as sample
                }
        except:
            pass
        
        # Return empty schema
        return {
            'table': table_name,
            'columns': [],
            'sample_data': []
        }
    
    async def get_grouped_data(self, request: TanStackRequest) -> TanStackResponse:
        """Handle grouped data request (for hierarchical tables)"""
        pivot_spec = self.convert_tanstack_request_to_pivot_spec(request)
        
        # Use the controller to get grouped data
        pivot_result = self.controller.run_pivot(pivot_spec, return_format="dict")
        
        # Format for TanStack grouping
        tanstack_result = self.convert_pivot_result_to_tanstack_format(pivot_result, request)
        
        # Add grouping information
        if request.grouping:
            tanstack_result.grouping = [{
                'id': group_col,
                'value': None  # Will be populated with actual grouped data
            } for group_col in request.grouping]
        
        return tanstack_result


# Utility function for TanStack integration
def create_tanstack_adapter(backend_uri: str = ":memory:") -> TanStackPivotAdapter:
    """Create a TanStack adapter with a configured controller"""
    controller = ScalablePivotController(
        backend_uri=backend_uri,
        enable_streaming=True,
        enable_incremental_views=True,
        tile_size=100,
        cache_ttl=300
    )
    return TanStackPivotAdapter(controller)


# Example usage functions
async def example_usage():
    """Example of how to use the TanStack adapter"""
    adapter = create_tanstack_adapter()
    
    # Example TanStack request
    request = TanStackRequest(
        operation=TanStackOperation.GET_DATA,
        table="sales",
        columns=[
            {"id": "region", "header": "Region", "enableSorting": True},
            {"id": "product", "header": "Product", "enableSorting": True},
            {"id": "total_sales", "header": "Total Sales", "aggregationFn": "sum", "aggregationField": "sales"}
        ],
        filters=[],
        sorting=[{"id": "total_sales", "desc": True}],
        grouping=["region", "product"],
        aggregations=[],
        pagination={"pageIndex": 0, "pageSize": 100}
    )
    
    result = await adapter.handle_request(request)
    print(f"Received {len(result.data)} rows from TanStack adapter")
    
    return result


if __name__ == "__main__":
    asyncio.run(example_usage())