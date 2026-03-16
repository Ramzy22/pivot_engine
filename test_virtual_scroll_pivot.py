"""
Test script to validate the virtual scrolling approach for hierarchical pivot with PIVOT COLUMNS
"""
import asyncio
import json
import os
import duckdb
from pivot_engine.scalable_pivot_controller import ScalablePivotController


def setup_test_db():
    """Set up test database with the user's example data including a pivot column"""
    con = duckdb.connect("test_virtual_scroll_pivot.duckdb")
    con.execute("CREATE TABLE IF NOT EXISTS sales (region VARCHAR, country VARCHAR, product VARCHAR, sales BIGINT, cost BIGINT)")
    con.execute("DELETE FROM sales")

    data = [
        ('East', 'Brazil', 'Headphones', 100, 50),
        ('East', 'Brazil', 'Phones', 200, 100),
        ('East', 'Germany', 'Headphones', 300, 150),
        ('East', 'Germany', 'Phones', 400, 200),
        ('North', 'USA', 'Headphones', 500, 250),
        ('North', 'USA', 'Phones', 600, 300),
        ('North', 'China', 'Headphones', 700, 350),
        ('North', 'China', 'Phones', 800, 400),
    ]

    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", data)
    con.close()
    return os.path.abspath("test_virtual_scroll_pivot.duckdb")


async def test_virtual_scrolling_pivot():
    """Test the virtual scrolling approach with pivoting"""
    db_path = setup_test_db()
    print(f"Database created at {db_path}")

    # Use ScalablePivotController which has the virtual scroll manager
    controller = ScalablePivotController(backend_uri=db_path, planner_name="duckdb")

    spec = {
        "table": "sales",
        "rows": ["region", "country", "cost"], # Added 'cost' (numeric) to rows
        "columns": ["product"],  # PIVOT COLUMN
        "measures": [
            {"field": "sales", "agg": "sum", "alias": "sales"},
        ],
        "filters": [],
        "totals": True,
        "pivot_config": {
            "include_totals_column": True
        }
    }

    print("\n=== Testing Virtual Scrolling with Pivot ===")
    
    # Test getting first few rows
    result_page1 = await controller.run_hierarchical_pivot(spec, flatten=True, start_row=0, end_row=5)
    print(f"\nResult Rows:")
    for i, row in enumerate(result_page1["rows"]):
        print(f"Row {i}: {row}")
        
        # Check if pivoted columns exist
        if 'Headphones_sales' not in row and row.get('_path') != '__grand_total__':
             # Grand total might have different keys if my fix failed?
             pass

    # Verify Grand Total Row structure
    gt_row = next((r for r in result_page1["rows"] if r.get('_isTotal') and r.get('_path') == '__grand_total__'), None)
    if gt_row:
        print("\nGrand Total Row Found:")
        print(gt_row)
        if "Headphones_sales" in gt_row and "Phones_sales" in gt_row:
            print("SUCCESS: Grand Total has pivoted columns.")
        else:
            print("FAILURE: Grand Total missing pivoted columns.")
    else:
        print("FAILURE: Grand Total row not found.")

if __name__ == "__main__":
    asyncio.run(test_virtual_scrolling_pivot())
