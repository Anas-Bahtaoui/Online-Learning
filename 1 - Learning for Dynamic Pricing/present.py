from dash.exceptions import PreventUpdate

import preamble

from dash import Dash, Input, Output, html, page_container, dcc, page_registry, State
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, pages_folder="web/pages", external_stylesheets=[dbc.themes.BOOTSTRAP])


@app.callback(Output("footer", "children"), Input("url", "pathname"))
def update_pagination(pathname):
    pages_ = list(page_registry.values())
    current_index = [page_index for page_index, page in enumerate(pages_) if page["path"] == pathname][0]
    return [
        dbc.Button("Previous", href=pages_[(current_index - 1) % len(pages_)]["path"], id="previous_page"),
        dbc.Button("Next", href=pages_[(current_index + 1) % len(pages_)]["path"], id="next_page"),
    ]


location = dcc.Location(id='url', refresh=False)
footer = dbc.Container([
    dbc.Row([
        dbc.Col([
        ], id="footer"),
    ])
])
app.layout = html.Div([
    location, page_container, footer
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
