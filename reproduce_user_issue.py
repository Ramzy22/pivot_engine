import duckdb
import json
import os
from pivot_engine.controller import PivotController

def setup_db():
    con = duckdb.connect("user_repro.duckdb")
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
    return os.path.abspath("user_repro.duckdb")

import asyncio

async def main():
    db_path = setup_db()
    print(f"Database created at {db_path}")
    
    controller = PivotController(backend_uri=db_path, planner_name="duckdb") # Using duckdb planner/backend
    
    spec = {
        "table": "sales",
        "rows": ["region", "country"],
        "measures": [
            {"field": "sales", "agg": "sum", "alias": "sales"},
            {"field": "cost", "agg": "sum", "alias": "cost"}
        ],
        "filters": [],
        "totals": True
    }
    
    print("Running hierarchical pivot...")
    try:
        # Initial run (top level)
        result = await controller.run_hierarchical_pivot(spec)
        print("\nTop Level Result (nested structure):")
        print(json.dumps(result["rows"], indent=2))

        spec_hash = result["spec_hash"]

        # Expand 'East'
        print("\nExpanding 'East'...")
        controller.toggle_expansion(spec_hash, ["East"])
        result_expanded = await controller.run_hierarchical_pivot(spec)
        print("\nExpanded Result (nested structure):")
        print(json.dumps(result_expanded["rows"], indent=2))

        # Now test the flattened version
        print("\nFlattened Result (flat list format):")
        result_flattened = await controller.run_hierarchical_pivot(spec, flatten=True)
        print(json.dumps(result_flattened["rows"], indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
