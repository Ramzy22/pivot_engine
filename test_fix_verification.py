"""
Test script to verify the fixes for the Dash expansion issue
"""
import asyncio
import json
import os
import duckdb
from pivot_engine.scalable_pivot_controller import ScalablePivotController
from pivot_engine.tanstack_adapter import TanStackPivotAdapter, TanStackRequest, TanStackOperation
from pivot_engine.types.pivot_spec import PivotSpec, Measure


def setup_test_db():
    """Set up test database with the user's example data"""
    con = duckdb.connect("test_fix_verification.duckdb")
    con.execute("CREATE TABLE IF NOT EXISTS sales (region VARCHAR, country VARCHAR, sales BIGINT, cost BIGINT)")
    con.execute("DELETE FROM sales")

    data = [
        ('East', 'Brazil', 124500000, 99500000),
        ('East', 'Germany', 125500000, 100500000),
        ('North', 'USA', 124000000, 99000000),
        ('North', 'China', 125000000, 100000000),
        ('South', 'Canada', 124250000, 99250000),
        ('South', 'Japan', 125250000, 100250000),
        ('West', 'France', 125750000, 100750000),
        ('West', 'UK', 124750000, 99750000),
    ]

    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", data)
    con.close()
    return os.path.abspath("test_fix_verification.duckdb")


async def test_fix_verification():
    """Test that the fixes work correctly"""
    db_path = setup_test_db()
    print(f"Database created at {db_path}")

    # Use ScalablePivotController which has full hierarchical support
    controller = ScalablePivotController(backend_uri=db_path, planner_name="duckdb")
    adapter = TanStackPivotAdapter(controller)

    # Create a TanStack request for hierarchical data
    request = TanStackRequest(
        operation=TanStackOperation.GET_DATA,
        table="sales",
        columns=[
            {"id": "region", "header": "Region", "accessorKey": "region"},
            {"id": "country", "header": "Country", "accessorKey": "country"},
            {"id": "sales", "header": "Sales", "accessorKey": "sales", "aggregationFn": "sum"},
            {"id": "cost", "header": "Cost", "accessorKey": "cost", "aggregationFn": "sum"}
        ],
        filters={},
        sorting=[{"id": "sales", "desc": True}],
        grouping=["region", "country"],  # This defines the hierarchy
        aggregations=[],
        totals=True
    )

    # 0. Create Materialized Hierarchy (Required for ScalablePivotController)
    print("\n0. Creating Materialized Hierarchy...")
    spec = PivotSpec(
        table="sales",
        rows=["region", "country"],
        measures=[
            Measure(field="sales", agg="sum", alias="sales"),
            Measure(field="cost", agg="sum", alias="cost")
        ],
        totals=True
    )
    controller.materialized_hierarchy_manager.create_materialized_hierarchy(spec)
    print("   Hierarchy created.")

    print("\n=== Testing Fix Verification ===")
    
    # Test hierarchical request with NO expansion (should only show regions)
    print("\n1. Testing with no expansion (collapsed state)...")
    collapsed_result = await adapter.handle_hierarchical_request(request, expanded_paths=[])
    print(f"   Result has {len(collapsed_result.data)} rows")
    
    # Check if only regions are shown (depth 0), no countries (depth 1)
    regions_only = all(row.get('depth', 0) == 0 for row in collapsed_result.data)
    countries_present = any(row.get('depth', 0) == 1 for row in collapsed_result.data)
    
    print(f"   Only regions shown (depth=0): {regions_only}")
    print(f"   Countries present (depth=1): {countries_present}")
    print(f"   Expected: regions_only=True, countries_present=False")
    
    if regions_only and not countries_present:
        print("   SUCCESS: COLLAPSED STATE WORKING CORRECTLY")
    else:
        print("   ERROR: COLLAPSED STATE NOT WORKING")

    # Show sample of collapsed data
    print("   Sample collapsed data:")
    for i, row in enumerate(collapsed_result.data[:5]):
        depth = row.get('depth', 'N/A')
        region = row.get('region', 'N/A')
        country = row.get('country', 'N/A')
        sales = row.get('sales', 'N/A')
        print(f"     Row {i}: depth={depth}, region='{region}', country='{country}', sales={sales}")

    # Test hierarchical request with East expanded
    print("\n2. Testing with 'East' expanded...")
    expanded_result = await adapter.handle_hierarchical_request(request, expanded_paths=[['East']])
    print(f"   Result has {len(expanded_result.data)} rows")

    # Check if regions and East's countries are shown
    has_regions = any(row.get('depth', 0) == 0 for row in expanded_result.data)
    has_east_countries = any(
        row.get('depth', 0) == 1 and row.get('region') == 'East'
        for row in expanded_result.data
    )

    print(f"   Has regions (depth=0): {has_regions}")
    print(f"   Has East's countries (depth=1 under East): {has_east_countries}")

    if has_regions and has_east_countries:
        print("   SUCCESS: EXPANSION WORKING CORRECTLY")
    else:
        print("   ERROR: EXPANSION NOT WORKING")

    # Show sample of expanded data
    print("   Sample expanded data:")
    for i, row in enumerate(expanded_result.data[:10]):  # Show more to see expansion
        depth = row.get('depth', 'N/A')
        region = row.get('region', 'N/A')
        country = row.get('country', 'N/A')
        sales = row.get('sales', 'N/A')
        print(f"     Row {i}: depth={depth}, region='{region}', country='{country}', sales={sales}")

    # Test 'Expand All' wildcard
    print("\n3. Testing with 'Expand All' wildcard [['__ALL__']]...")
    expand_all_result = await adapter.handle_hierarchical_request(request, expanded_paths=[['__ALL__']])
    print(f"   Result has {len(expand_all_result.data)} rows")

    # In expand all, all regions and all countries should be present
    all_depth_0 = [row for row in expand_all_result.data if row.get('depth') == 0]
    all_depth_1 = [row for row in expand_all_result.data if row.get('depth') == 1]
    
    print(f"   Number of top-level rows: {len(all_depth_0)}")
    print(f"   Number of child rows: {len(all_depth_1)}")
    
    if len(all_depth_1) > 0:
        print("   SUCCESS: EXPAND ALL WORKING CORRECTLY")
    else:
        print("   ERROR: EXPAND ALL NOT WORKING")

    print("\n=== Fix Verification Complete ===")
    if regions_only and not countries_present:
        print("SUCCESS: Initial collapsed state works correctly!")
        print("SUCCESS: Only top-level items shown initially!")
    else:
        print("ERROR: Initial state still shows all expanded!")


if __name__ == "__main__":
    asyncio.run(test_fix_verification())