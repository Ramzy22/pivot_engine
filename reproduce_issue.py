
import dash
from dash import html
import dash_tanstack_pivot
import pandas as pd

app = dash.Dash(__name__)

df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 20, 15, 25, 12]
})

data = df.to_dict('records')

app.layout = html.Div([
    dash_tanstack_pivot.DashTanstackPivot(
        id='pivot',
        data=data,
        rowFields=['Category'],
        valConfigs=[{'field': 'Value', 'agg': 'sum'}],
        columnPinning={'left': [], 'right': []},
        rowPinning={'top': [], 'bottom': []},
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
