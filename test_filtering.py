import duckdb
import ibis
from pivot_engine.common.ibis_expression_builder import IbisExpressionBuilder
import pyarrow as pa

def test_filtering():
    # Setup
    con = ibis.duckdb.connect(":memory:")
    
    # Create via SQL or Arrow to ensure schema
    con.con.execute("CREATE TABLE sales (sales BIGINT, region VARCHAR)")
    con.con.execute("INSERT INTO sales VALUES (100, 'North'), (249, 'South'), (300, 'East')")
    
    t = con.table("sales")
    
    builder = IbisExpressionBuilder(con)
    
    print("--- Test 1: Exact Match Numeric '249' ---")
    # This simulates what TanStack might send if type is 'eq'
    filters = [{"field": "sales", "op": "eq", "value": "249"}]
    expr = builder.build_filter_expression(t, filters)
    result = t.filter(expr).execute()
    print(f"Result rows: {len(result)}")
    assert len(result) == 1
    assert result['sales'][0] == 249

    print("\n--- Test 2: Contains '249' on Numeric ---")
    # This simulates 'contains' filter on numeric column
    filters = [{"field": "sales", "op": "contains", "value": "249"}]
    expr = builder.build_filter_expression(t, filters)
    result = t.filter(expr).execute()
    print(f"Result rows: {len(result)}")
    assert len(result) == 1
    assert result['sales'][0] == 249

    print("\n--- Test 3: Contains '24' on Numeric ---")
    filters = [{"field": "sales", "op": "contains", "value": "24"}]
    expr = builder.build_filter_expression(t, filters)
    result = t.filter(expr).execute()
    print(f"Result rows: {len(result)}")
    assert len(result) == 1
    assert result['sales'][0] == 249

    print("\nSUCCESS")

if __name__ == "__main__":
    test_filtering()