"""
app.py - Enterprise Grade Server-Side Pivot Table
Integrates DashTanstackPivot with ScalablePivotEngine.
"""
import os
import sys

# Add the parent directory's pivot_engine folder to sys.path to ensure we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pivot_engine')))

import pyarrow as pa
from dash import Dash, html, dcc, dash_table
from flask import request as flask_request, jsonify

from dash_tanstack_pivot import DashTanstackPivot
from pivot_engine import create_tanstack_adapter
from pivot_engine.runtime import (
    PivotRuntimeService,
    SessionRequestGate,
    register_dash_drill_modal_callback,
    register_dash_pivot_transport_callback,
)

_adapter = None
_runtime_service = None
_DEBUG_OUTPUT = os.environ.get("PIVOT_DEBUG_OUTPUT", "1").lower() in {"1", "true", "yes"}

_SESSION_GATE = SessionRequestGate()


def get_runtime_service():
    global _runtime_service
    if _runtime_service is None:
        _runtime_service = PivotRuntimeService(
            adapter_getter=get_adapter,
            session_gate=_SESSION_GATE,
            debug=_DEBUG_OUTPUT,
        )
    return _runtime_service
# --- 2. Data Loading (Simulation) ---
def load_initial_data(adapter):
    if _DEBUG_OUTPUT:
        print("Generating simulation data (2M rows)...")
    # Generate 2M rows for stress testing
    rows = 2000000 
    
    # Create more diverse date range for column virtualization test
    dates = [f"2023-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29, 2)] # ~150 unique dates
    
    data_source = {
        "region": (["North", "South", "East", "West"] * (rows // 4)),
        "country": (["USA", "Canada", "Brazil", "UK", "China", "Japan", "Germany", "France"] * (rows // 8)),
        "product": (["Laptop", "Phone", "Tablet", "Monitor", "Headphones"] * (rows // 5)),
        "sales": [x % 1000 for x in range(rows)],
        "cost": [x % 800 for x in range(rows)],
        "date": (dates * (rows // len(dates)) + dates[:rows % len(dates)])
    }
    table = pa.Table.from_pydict(data_source)
    
    # Load into the engine
    adapter.controller.load_data_from_arrow("sales_data", table)
    if _DEBUG_OUTPUT:
        print(f"Data loaded into Pivot Engine: {rows} rows.")
    
    # Pre-materialize hierarchy for the default view to ensure virtual scroll works immediately
    from pivot_engine.types.pivot_spec import PivotSpec, Measure
    default_spec = PivotSpec(
        table="sales_data",
        rows=["region", "country"],
        measures=[
            Measure(field="sales", agg="sum", alias="sales_sum"),
            Measure(field="cost", agg="sum", alias="cost_sum")
        ]
    )
    if _DEBUG_OUTPUT:
        print("Pre-materializing default hierarchy...")
    adapter.controller.materialized_hierarchy_manager.create_materialized_hierarchy(default_spec)
    if _DEBUG_OUTPUT:
        print("Hierarchy materialized.")


def get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = create_tanstack_adapter(backend_uri=":memory:")
        load_initial_data(_adapter)
    return _adapter

# --- 3. Dash App ---
app = Dash(__name__)


@app.server.route('/api/drill-through')
def api_drill_through():
    import asyncio
    from pivot_engine.types.pivot_spec import PivotSpec

    table = flask_request.args.get('table', '')
    row_path = flask_request.args.get('row_path', '')
    row_fields_raw = flask_request.args.get('row_fields', '')
    page = int(flask_request.args.get('page', 0))
    page_size = min(int(flask_request.args.get('page_size', 500)), 500)
    sort_col = flask_request.args.get('sort_col') or None
    sort_dir = flask_request.args.get('sort_dir', 'asc')
    text_filter = flask_request.args.get('filter', '')

    if not table:
        return jsonify({'error': 'table param required'}), 400

    row_fields = [f for f in row_fields_raw.split(',') if f]
    path_parts = row_path.split('|||') if row_path else []

    drill_filters = []
    for i, field in enumerate(row_fields):
        if i < len(path_parts) and path_parts[i]:
            drill_filters.append({'field': field, 'op': '=', 'value': path_parts[i]})

    spec = PivotSpec(table=table, rows=[], measures=[], filters=[])
    result = asyncio.run(
        get_adapter().controller.get_drill_through_data(
            spec,
            drill_filters,
            limit=page_size,
            offset=page * page_size,
            sort_col=sort_col,
            sort_dir=sort_dir,
            text_filter=text_filter,
        )
    )
    return jsonify({
        'rows': result['rows'],
        'page': page,
        'page_size': page_size,
        'total_rows': result['total_rows'],
    })


app.layout = html.Div([
    dcc.Store(id="drill-data-store"),
    html.Div(
        DashTanstackPivot(
            id="pivot-grid",
            style={"height": "800px", "width": "100%"},
            table="sales_data",
            # Enable Server Side Mode
            serverSide=True,
            # Initial Configuration
            rowFields=["region", "country"],
            colFields=[],
            valConfigs=[{"field": "sales", "agg": "sum"}, {"field": "cost", "agg": "sum"}],
            filters={},
            sorting=[],
            expanded={},
            # Pass ALL available fields as columns definition for the sidebar
            columns=[
                {"id": "region"}, {"id": "country"}, {"id": "product"}, 
                {"id": "sales"}, {"id": "cost"}, {"id": "date"}
            ],
            availableFieldList=["region", "country", "product", "sales", "cost", "date"],
            # Initial Data (Empty, will fetch on load)
            data=[],
            rowCount=0,
            filterOptions={},
            validationRules={
                "sales_sum": [{"type": "numeric"}, {"type": "min", "value": 0}],
                "cost_sum": [{"type": "numeric"}, {"type": "min", "value": 0}]
            }
        ),
        style={'padding': '0 16px'}
    ),

    # Drill Through Modal
    html.Div(id="drill-modal", children=[
        html.Div([
            html.Div([
                html.H2("Drill Through: Raw Records", style={'margin': 0}),
                html.Button("X", id="close-drill", style={
                    'border': 'none', 'background': 'none', 'fontSize': '20px', 'cursor': 'pointer'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '16px'}),
            
            html.Div(id="drill-table-container", children=[
                dash_table.DataTable(
                    id="drill-table",
                    columns=[],
                    data=[],
                    page_size=15,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '13px'},
                    style_header={'backgroundColor': '#f5f5f5', 'fontWeight': 'bold'}
                )
            ])
        ], style={
            'position': 'relative', 'margin': '5% auto', 'padding': '20px', 
            'width': '80%', 'backgroundColor': '#fff', 'borderRadius': '8px',
            'boxShadow': '0 4px 20px rgba(0,0,0,0.2)', 'maxHeight': '80vh', 'overflowY': 'auto'
        })
    ], style={
        'display': 'none', 'position': 'fixed', 'zIndex': 10002, 'left': 0, 'top': 0, 
        'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)'
    })
])

# --- 4. Reusable Runtime Callback Wiring ---
register_dash_pivot_transport_callback(
    app,
    get_runtime_service,
    pivot_id="pivot-grid",
    drill_store_id="drill-data-store",
    debug=_DEBUG_OUTPUT,
)

register_dash_drill_modal_callback(
    app,
    drill_store_id="drill-data-store",
    close_drill_id="close-drill",
    drill_modal_id="drill-modal",
    drill_table_id="drill-table",
)

if __name__ == "__main__":
    app.run(debug=True, port=8050)


