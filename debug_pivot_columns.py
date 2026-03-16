
import asyncio
import os
import duckdb
import pyarrow as pa
from pivot_engine.scalable_pivot_controller import ScalablePivotController
from pivot_engine.tanstack_adapter import TanStackPivotAdapter, TanStackRequest, TanStackOperation

def create_test_db():
    db_path = "debug_pivot.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE sales (product VARCHAR, sales INTEGER, cost INTEGER)")
    con.execute("INSERT INTO sales VALUES ('A', 100, 50), ('B', 200, 100)")
    con.close()
    return db_path

async def test_pivot():
    db_path = create_test_db()
    controller = ScalablePivotController(backend_uri=db_path)
    adapter = TanStackPivotAdapter(controller)
    
    # Request with 2 measures (Sales, Cost) pivoted by Product
    request = TanStackRequest(
        operation=TanStackOperation.GET_DATA,
        table="sales",
        columns=[
            # Measures
            {"id": "sales_sum", "aggregationField": "sales", "aggregationFn": "sum"},
            {"id": "cost_sum", "aggregationField": "cost", "aggregationFn": "sum"},
            # Pivot Dimension
            {"id": "product"}
        ],
        filters=[],
        sorting=[],
        grouping=[], # No row grouping, just pivot columns
        aggregations=[],
        pagination={"pageIndex": 0, "pageSize": 100}
    )
    
    # We need to simulate how app.py calls it.
    # In app.py: colFields=['product']. grouping=rowFields (empty in this test case for simplicity)
    # convert_tanstack_request_to_pivot_spec puts 'product' in columns if not grouped.
    
    print("Converting request...")
    spec = adapter.convert_tanstack_request_to_pivot_spec(request)
    print(f"Spec measures: {[m.alias for m in spec.measures]}")
    print(f"Spec columns: {spec.columns}")
    
    print("\nRunning pivot...")
    result = await controller.run_pivot_async(spec, return_format="dict")
    
    print(f"\nResult columns: {result['columns']}")
    print(f"Result rows: {result['rows']}")
    
    # Check if both measures exist for product A
    expected_cols = ['A_sales_sum', 'A_cost_sum', 'B_sales_sum', 'B_cost_sum']
    missing = [c for c in expected_cols if c not in result['columns']]
    
    if missing:
        print(f"\n[FAIL] Missing columns: {missing}")
    else:
        print(f"\n[PASS] All expected columns found.")

    controller.close()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_pivot())
