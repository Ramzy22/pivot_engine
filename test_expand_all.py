"""
Test script to validate the 'Expand All' wildcard logic in hierarchical_scroll_manager.py
"""
import asyncio
import os
import duckdb
from pivot_engine.scalable_pivot_controller import ScalablePivotController
from pivot_engine.tanstack_adapter import TanStackPivotAdapter, TanStackRequest, TanStackOperation


def setup_test_db():
    """Set up test database with multi-level hierarchy"""
    db_path = "test_expand_all.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE sales (region VARCHAR, country VARCHAR, city VARCHAR, sales BIGINT)")

    data = [
        ('East', 'Germany', 'Berlin', 100),
        ('East', 'Germany', 'Munich', 150),
        ('East', 'France', 'Paris', 200),
        ('West', 'USA', 'New York', 300),
        ('West', 'USA', 'LA', 350),
        ('West', 'Canada', 'Toronto', 400),
    ]

    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", data)
    con.close()
    return os.path.abspath(db_path)


async def test_expand_all():
    """Test the 'Expand All' functionality"""
    db_path = setup_test_db()
    print(f"Database created at {db_path}")

    controller = ScalablePivotController(backend_uri=db_path, planner_name="duckdb")
    adapter = TanStackPivotAdapter(controller)

    # Create a TanStack request for 3-level hierarchy
    request = TanStackRequest(
        operation=TanStackOperation.GET_DATA,
        table="sales",
        columns=[
            {"id": "region", "header": "Region", "accessorKey": "region"},
            {"id": "country", "header": "Country", "accessorKey": "country"},
            {"id": "city", "header": "City", "accessorKey": "city"},
            {"id": "sales", "header": "Sales", "accessorKey": "sales", "aggregationFn": "sum"}
        ],
        filters={},
        sorting=[{"id": "region", "desc": False}],
        grouping=["region", "country", "city"],
        aggregations=[],
        totals=True
    )

    print("\n=== Testing 'Expand All' Wildcard Logic ===")
    
    # 1. No expansion
    res1 = await adapter.handle_hierarchical_request(request, expanded_paths=[])
    print(f"No expansion: {len(res1.data)} rows")
    # Expected: 2 regions + 1 grand total = 3 rows
    for row in res1.data:
        print(f"  depth={row.get('depth')}, region={row.get('region')}, country={row.get('country')}, city={row.get('city')}")

    # 2. Expand All wildcard [['__ALL__']]
    print("\nTriggering Expand All with [['__ALL__']]...")
    res2 = await adapter.handle_hierarchical_request(request, expanded_paths=[['__ALL__']])
    print(f"Expand All: {len(res2.data)} rows")
    
    # Expected:
    # Grand Total (depth 0, or None depending on implementation, usually it's at the end or top)
    # Region East (depth 0)
    #   Germany (depth 1)
    #     Berlin (depth 2)
    #     Munich (depth 2)
    #   France (depth 1)
    #     Paris (depth 2)
    # Region West (depth 0)
    #   USA (depth 1)
    #     New York (depth 2)
    #     LA (depth 2)
    #   Canada (depth 1)
    #     Toronto (depth 2)
    # Total rows = 2 regions + 4 countries + 6 cities + 1 grand total = 13 rows
    
    for i, row in enumerate(res2.data):
        depth = row.get('depth')
        region = row.get('region')
        country = row.get('country')
        city = row.get('city')
        print(f"  Row {i}: depth={depth}, region={region}, country={country}, city={city}")

    # Verify that we have depth 2 rows (leaves)
    has_leaves = any(row.get('depth') == 2 for row in res2.data)
    if has_leaves:
        print("\nSUCCESS: 'Expand All' correctly expanded to leaf levels!")
    else:
        print("\nFAILURE: 'Expand All' did not expand to leaf levels.")

    # 3. Verify specific expansion still works
    print("\nTesting specific expansion of 'East'...")
    res3 = await adapter.handle_hierarchical_request(request, expanded_paths=[['East']])
    print(f"Expand 'East': {len(res3.data)} rows")
    # Expected: Region East (exp), Region West (coll), Grand Total, Germany (coll), France (coll) = 3 + 2 = 5 rows
    # Wait: 
    # East (depth 0, expanded)
    #   Germany (depth 1, collapsed)
    #   France (depth 1, collapsed)
    # West (depth 0, collapsed)
    # Grand Total
    # Total = 5 rows
    for row in res3.data:
         print(f"  depth={row.get('depth')}, region={row.get('region')}, country={row.get('country')}")

    if len(res2.data) > len(res3.data):
        print("\nAll tests passed!")
    else:
        print("\nSomething is wrong with row counts.")


if __name__ == "__main__":
    asyncio.run(test_expand_all())
