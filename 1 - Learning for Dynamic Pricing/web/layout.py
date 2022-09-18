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

experiment_count_selector = dbc.Input(
    id=ids.experiment_count,
    type="number",
    min=0,
    max=1000,
    value=1,
)

run_experiment_button = dbc.Button("Run Experiment", id=ids.run_experiment, n_clicks=0, color="primary")
reset_results_button = dbc.Button("Reset Results", id=ids.reset_results, n_clicks=0, color="primary")

resolution_selector = dcc.Dropdown(
    id=ids.resolution_selector,
    searchable=False,
    clearable=False,
    value=1,
    options=[
        {"label": "1", "value": 1},
        {"label": "5", "value": 5},
        {"label": "10", "value": 10},
        {"label": "20", "value": 20},
    ])
hidden = {"display": "none"}
visible = {"display": "block"}

default_aggregate_is_selected = False
shall_aggregate_selector = dbc.Switch(id=ids.experiment_toggle, value=default_aggregate_is_selected)
aggregate_function_selector = dcc.Dropdown(
    id=ids.experiment_aggregate_selector,
    searchable=False,
    clearable=False,
    value="mean",
    disabled=not default_aggregate_is_selected,
    options=[
        {"label": "Mean", "value": "mean"},
        {"label": "Max", "value": "max"},
        {"label": "Min", "value": "min"},
    ])
experiment_day_selector = dbc.Input(
    id=ids.experiment_day_selector,
    type="number",
    min=0,
    max=0,
    value=0,
    disabled=default_aggregate_is_selected,
    style=hidden if default_aggregate_is_selected else visible,
    step=1,

)
second_row = dbc.Row([
    dbc.Col([
        html.P("How many days to run each experiment: "),
        run_count_selector,
    ], md=3),
    dbc.Col([
        html.P("How many experiments to run: "),
        experiment_count_selector,
    ], md=3),
    dbc.Col([
        run_experiment_button,
        reset_results_button
    ], md=3),
])
third_row = dbc.Row([
    dbc.Col([
        html.P("Result Resolution: "),
        resolution_selector,
    ], md=3),
    dbc.Col([
        html.P("Aggregate results? "),
        shall_aggregate_selector,
    ], md=3),
    dbc.Col([
        html.P("Aggregate function: "),
        aggregate_function_selector,
    ], md=3,
        style=visible if default_aggregate_is_selected else hidden,
        id="exp_agg_selector",
    ),
    dbc.Col([
        html.P("Show Experiment i: "),
        experiment_day_selector,
    ], md=3,
        style=visible if not default_aggregate_is_selected else hidden, id="exp_day_selector"
    ),
    dbc.Col([
        html.P("Customer Day: "),
        dbc.Input(id=ids.customer_day_selector, type="number", min=0, max=1000, value=0),
    ], md=3, id="customer_day_col"),
], id="third_row", style=hidden)

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
    third_row,
    results,
])
