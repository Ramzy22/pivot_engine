import dash
from dash import html
import dash_tanstack_pivot
import pandas as pd

# Minimal data for debugging
data = [
    {"Region": "North", "Country": "USA", "Product": "A", "Sales": 100},
    {"Region": "North", "Country": "USA", "Product": "B", "Sales": 150},
    {"Region": "South", "Country": "Brazil", "Product": "A", "Sales": 200},
    {"Region": "South", "Country": "Brazil", "Product": "B", "Sales": 250},
]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Debug Pivot"),
    dash_tanstack_pivot.DashTanstackPivot(
        id="pivot",
        data=data,
        rowFields=["Region", "Country"],
        colFields=["Product"],
        valConfigs=[{"field": "Sales", "agg": "sum"}],
        style={"height": "600px", "width": "100%"}
    )
])

if __name__ == "__main__":
    app.run_server(debug=True, port=8055)