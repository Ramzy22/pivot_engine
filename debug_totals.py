
import asyncio
import os
import duckdb
import pyarrow as pa
from pivot_engine.scalable_pivot_controller import ScalablePivotController
from pivot_engine.tanstack_adapter import TanStackPivotAdapter, TanStackRequest, TanStackOperation
from pivot_engine.types.pivot_spec import PivotConfig

def create_test_db():
    db_path = "debug_totals.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE sales (region VARCHAR, year INTEGER, sales INTEGER)")
    con.execute("INSERT INTO sales VALUES ('A', 2024, 100), ('A', 2024, 200), ('B', 2024, 300)")
    con.close()
    return db_path

async def test_totals():
    db_path = create_test_db()
    controller = ScalablePivotController(backend_uri=db_path)
    adapter = TanStackPivotAdapter(controller)
    
    # Request with Pivoting AND Totals
    # Using handle_hierarchical_request as app.py does
    
    request = TanStackRequest(
        operation=TanStackOperation.GET_DATA,
        table="sales",
        columns=[
            {"id": "sales_sum", "aggregationField": "sales", "aggregationFn": "sum"},
            {"id": "year"}
        ],
        filters=[],
        sorting=[],
        grouping=["region"],
        aggregations=[],
        pagination={"pageIndex": 0, "pageSize": 100}
    )
    
    print("\n--- Testing Hierarchy with Totals ---")
    
    # We need to manually construct PivotSpec to ensure totals=True is set?
    # No, adapter does it.
    # Adapter sets totals=True by default in convert_tanstack_request_to_pivot_spec.
    
    spec = adapter.convert_tanstack_request_to_pivot_spec(request)
    print(f"Spec Totals: {spec.totals}")
    print(f"Spec Pivot Config: {spec.pivot_config}")
    
    # Run Hierarchical Request
    response = await adapter.handle_hierarchical_request(request, expanded_paths=[])
    
    print(f"\nResponse Rows ({len(response.data)}):")
    total_found = False
    for row in response.data:
        print(row)
        if row.get('_isTotal'):
            total_found = True
            
    if total_found:
        print("\n[PASS] Grand Total row found.")
    else:
        print("\n[FAIL] Grand Total row NOT found.")

    controller.close()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_totals())
