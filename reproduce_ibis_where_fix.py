import ibis
import pandas as pd
from pivot_engine.planner.ibis_planner import IbisPlanner
from pivot_engine.types.pivot_spec import PivotSpec, Measure, PivotConfig

# Setup
con = ibis.duckdb.connect()
df = pd.DataFrame({
    'region': ['A', 'A', 'B', 'B'],
    'product': ['X', 'Y', 'X', 'Y'],
    'sales': [10, 20, 30, 40]
})
con.create_table('sales', df)

planner = IbisPlanner(con)

# Mimic the failing call in build_pivot_query_from_columns
spec = PivotSpec(
    table='sales',
    rows=['region'],
    columns=['product'],
    measures=[Measure(field='sales', agg='sum', alias='total_sales')]
)
column_values = ['X', 'Y']

try:
    print("Testing build_pivot_query_from_columns...")
    query = planner.build_pivot_query_from_columns(spec, column_values)
    print("Query built successfully!")
    print(con.execute(query))
except Exception as e:
    print(f"Caught expected error: {e}")
    import traceback
    traceback.print_exc()
