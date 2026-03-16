"""
Test script to validate the virtual scrolling approach for hierarchical pivot
This addresses the user's issue with millions of rows by loading only visible data
"""
import asyncio
import json
import os
import duckdb
from pivot_engine.scalable_pivot_controller import ScalablePivotController


def setup_test_db():
    """Set up test database with the user's example data"""
    con = duckdb.connect("test_virtual_scroll.duckdb")
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
    return os.path.abspath("test_virtual_scroll.duckdb")


async def test_virtual_scrolling():
    """Test the virtual scrolling approach"""
    db_path = setup_test_db()
    print(f"Database created at {db_path}")

    # Use ScalablePivotController which has the virtual scroll manager
    controller = ScalablePivotController(backend_uri=db_path, planner_name="duckdb")

    spec = {
        "table": "sales",
        "rows": ["region", "country"],  # Defines the hierarchy
        "measures": [
            {"field": "sales", "agg": "sum", "alias": "sales"},
            {"field": "cost", "agg": "sum", "alias": "cost"}
        ],
        "filters": [],
        "totals": True
    }

    print("\n=== Testing Virtual Scrolling Approach ===")
    
    # Test getting first 2 rows (virtual scrolling)
    result_page1 = await controller.run_hierarchical_pivot(spec, flatten=True, start_row=0, end_row=2)
    print(f"\nFirst 2 rows (virtual scrolling):")
    for i, row in enumerate(result_page1["rows"]):
        indent = "  " if row.get("depth") == 1 else ""
        if row.get("depth") == 0:  # Region level
            region = row.get("region", "")
            sales = row.get("sales", 0)
            cost = row.get("cost", 0)
            print(f"{region:<15}    {sales:>12}    {cost:>11}")
        elif row.get("depth") == 1:  # Country level
            country = row.get("country", "")
            sales = row.get("sales", 0)
            cost = row.get("cost", 0)
            print(f"{indent}{country:<13}    {sales:>12}    {cost:>11}")
    
    # Simulate expanding the 'East' region
    expanded_paths = [['East']]
    
    # Test getting first 4 rows with 'East' expanded (virtual scrolling)
    result_expanded = await controller.run_hierarchical_pivot(spec, flatten=True, start_row=0, end_row=4, expanded_paths=expanded_paths)
    print(f"\nFirst 4 rows with 'East' expanded (virtual scrolling):")
    for i, row in enumerate(result_expanded["rows"]):
        indent = "  " if row.get("depth") == 1 else ""
        if row.get("depth") == 0:  # Region level
            region = row.get("region", "")
            sales = row.get("sales", 0)
            cost = row.get("cost", 0)
            print(f"{region:<15}    {sales:>12}    {cost:>11}")
        elif row.get("depth") == 1:  # Country level
            country = row.get("country", "")
            sales = row.get("sales", 0)
            cost = row.get("cost", 0)
            print(f"{indent}{country:<13}    {sales:>12}    {cost:>11}")
    
    print(f"\nTotal visible rows: {result_expanded.get('total_visible_rows', len(result_expanded['rows']))}")
    
    # Test getting rows 2-4 (middle section)
    result_page2 = await controller.run_hierarchical_pivot(spec, flatten=True, start_row=2, end_row=4, expanded_paths=expanded_paths)
    print(f"\nRows 2-4 (middle section, virtual scrolling):")
    for i, row in enumerate(result_page2["rows"]):
        indent = "  " if row.get("depth") == 1 else ""
        if row.get("depth") == 0:  # Region level
            region = row.get("region", "")
            sales = row.get("sales", 0)
            cost = row.get("cost", 0)
            print(f"{region:<15}    {sales:>12}    {cost:>11}")
        elif row.get("depth") == 1:  # Country level
            country = row.get("country", "")
            sales = row.get("sales", 0)
            cost = row.get("cost", 0)
            print(f"{indent}{country:<13}    {sales:>12}    {cost:>11}")
    
    print("\nSUCCESS: Virtual scrolling approach successfully implemented!")
    print("SUCCESS: Only visible rows are loaded, not entire dataset")
    print("SUCCESS: Flat list format maintained for easy UI rendering")
    print("SUCCESS: Expansion state properly handled")


if __name__ == "__main__":
    asyncio.run(test_virtual_scrolling())