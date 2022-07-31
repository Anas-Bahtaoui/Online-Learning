from common import ids
from dash import dcc, html
import dash_bootstrap_components as dbc

from production import RUN_COUNT
from web.params import potentials

data_storage = dcc.Store(id=ids.storage, storage_type="memory", data={
    "clicks": 0,
    "results": None,
    "run_count": RUN_COUNT,
    "reset_clicks": 0,
})

# Config widgets
config_selectors = dbc.Row([
    dbc.Col([
        dbc.Label(k + ": "),
        dcc.Dropdown(
            id=f"{i}-selector",
            options=[{"label": x, "value": x} for x in v.keys()],
            value=list(v.keys())[0],
            searchable=False,
            clearable=False,
        )], width=3
    ) for i, (k, v) in enumerate(potentials.items(), )], className="mb-3")

run_count_selector = dbc.Input(
    id=ids.run_count,
    type="number",
    min=0,
    max=1000,
    value=RUN_COUNT,
)

run_experiment_button = dbc.Button("Run Experiment", id=ids.run_experiment, n_clicks=0, color="primary")
reset_results_button = dbc.Button("Reset Results", id=ids.reset_results, n_clicks=0, color="primary")

second_row = dbc.Row([
    dbc.Col([
        html.P("How many days to run: "),
        run_count_selector,
    ]),
    dbc.Col([
        run_experiment_button,
        reset_results_button
    ]),
])

dropdownProps = dict(
    searchable=False,
    clearable=False,
    placeholder="(no learner data)",
    disabled=True,
)
experiment_selector_left = dcc.Dropdown(
    id=ids.left_experiment_selector,
    **dropdownProps,
)
experiment_selector_right = dcc.Dropdown(
    id=ids.right_experiment_selector,
    **dropdownProps,
)

results = dbc.Row([
    dbc.Col([
        dbc.Row([experiment_selector_left]),
        dbc.Row(id=ids.result_div_left),
    ], xs=6),
    dbc.Col([
        dbc.Row([experiment_selector_right]),
        dbc.Row(id=ids.result_div_right),

    ], xs=6)
], id="results")

layout = dbc.Container([
    data_storage,
    config_selectors,
    second_row,
    results,
])
