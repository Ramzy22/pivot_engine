import ibis
import pandas as pd

# Setup
con = ibis.duckdb.connect()
t = con.create_table('t', pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))

try:
    expr = (t.a == 1).ifelse(t.b, ibis.null())
    print("ifelse works")
    print(con.execute(expr.name('res')))
except Exception as e:
    print(f"ifelse failed: {e}")
