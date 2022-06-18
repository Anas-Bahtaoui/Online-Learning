from typing import List
import preamble
import plotly.express
import plotly.graph_objs as go
import scipy.ndimage
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from Learner import PriceIndexes, ProductRewards
from entities import Dirichlet, CustomerTypeBased, PositiveIntegerGaussian as PIG, SimulationConfig, Simulation, Product
from production import LAMBDA_, product_configs, purchase_amounts, customer_counts, dirichlets, secondaries, \
    learners

dirichlet = Dirichlet([100, 100, 100, 100, 100, 100])
same_diriclets: CustomerTypeBased[Dirichlet] = CustomerTypeBased(
    professional=dirichlet,
    young_beginner=dirichlet,
    old_beginner=dirichlet,
)

## TODO: Unnecessary, we have constant distribution :) (but hope nothing uses private values :()
small_variance = 0.0001
potential_diriclets = {
    "Production": dirichlets,
    "Same Weights": same_diriclets,
}
zeros = CustomerTypeBased(
    professional=PIG(0, small_variance),
    young_beginner=PIG(0, small_variance),
    old_beginner=PIG(0, small_variance),
)
potential_customer_counts = {
    "Production": customer_counts,
    "Only Young": zeros._replace(young_beginner=customer_counts.young_beginner),
    "Only Old": zeros._replace(old_beginner=customer_counts.old_beginner),
    "Only Professional": zeros._replace(professional=customer_counts.professional),
    "Fixed": CustomerTypeBased(
        professional=PIG(customer_counts.professional.get_expectation(), small_variance),
        young_beginner=PIG(customer_counts.young_beginner.get_expectation(), small_variance),
        old_beginner=PIG(customer_counts.old_beginner.get_expectation(), small_variance),
    )
}

potential_purchase_amounts = {
    "Production": purchase_amounts,
    "All One": CustomerTypeBased(
        professional=[PIG(1, small_variance) for _ in range(5)],
        young_beginner=[PIG(1, small_variance) for _ in range(5)],
        old_beginner=[PIG(1, small_variance) for _ in range(5)],
    ),
    "Fixed": CustomerTypeBased(
        professional=[PIG(purchase_amounts.professional, small_variance) for _ in range(5)],
        young_beginner=[PIG(purchase_amounts.young_beginner, small_variance) for _ in range(5)],
        old_beginner=[PIG(purchase_amounts.old_beginner, small_variance) for _ in range(5)],
    ),
}
zero_secondaries = [[0 for _ in range(5)] for _ in range(5)]
potential_secondaries = {
    "Production": secondaries,
    "Disabled": CustomerTypeBased(
        professional=zero_secondaries,
        young_beginner=zero_secondaries,
        old_beginner=zero_secondaries,
    ),
}

potentials = {
    "Dirichlet value preset": potential_diriclets,
    "Customer count preset": potential_customer_counts,
    "Purchase amount preset": potential_purchase_amounts,
    "Secondary graph preset": potential_secondaries,
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

### Intermediate storage
data_storage = dcc.Store(id="data-storage", storage_type="local", data={
    "clicks": 0,
    "results": None,
    "run_count": 20,
})

# Config widgets
config_selectors = dbc.Row([
    dbc.Col([
        dbc.Label(k + ": "),
        dcc.Dropdown(
            id=f"{i}-selector",
            options=[{"label": x, "value": x} for x in v.keys()],
            value=list(v.keys())[0],
        )], width=3
    ) for i, (k, v) in enumerate(potentials.items(), )], className="mb-3")

run_count_selector = dbc.Input(
    id="run-count-selector",
    type="number",
    min=0,
    max=1000,
    value=20,
)

## Run experiment
run_experiment_button = dbc.Button("Run Experiment", id="run-experiment-button", n_clicks=0, color="primary")

second_row = dbc.Row([
    dbc.Col([
        html.P("How many days to run: "),
        run_count_selector,
    ]),
    dbc.Col([
        run_experiment_button
    ]),
])


@app.callback(
    Output(data_storage, "data"),
    Input(run_experiment_button, "n_clicks"),
    State(data_storage, "data"),
    State(run_count_selector, "value"),
    [State(f"{i}-selector", "value") for i in range(4)],
)
def run_experiment(clicks, saved_data, run_count, *config_values):
    learners_ = learners[:2]
    if saved_data["clicks"] == clicks:
        raise PreventUpdate
    saved_data["clicks"] = clicks
    config = SimulationConfig(
        dirichlets=potential_diriclets[config_values[0]],
        customer_counts=potential_customer_counts[config_values[1]],
        purchase_amounts=potential_purchase_amounts[config_values[2]],
        secondaries=potential_secondaries[config_values[3]],
        lambda_=LAMBDA_,
        product_configs=product_configs,
    )
    simulation = Simulation(config, learners_)
    simulation.run(run_count, log=False, plot_graphs=False, verbose=False)
    output_data = {}
    for learner in learners_:
        output_data[learner.name] = {"exp": learner._experiment_history,
                                     "products": [product.serialize() for product in simulation.products]}
        if hasattr(learner, "_customer_history"):
            output_data[learner.name]["customers"] = [[customer.serialize() for customer in day] for day in
                                                      learner._customer_history]
    saved_data["results"] = output_data
    return saved_data


# Result Graphs


def render_rewards(name, rewards):
    plot = plotly.express.line(x=range(1, len(rewards) + 1), y=scipy.ndimage.uniform_filter1d(rewards, size=10),
                               labels={"x": "Iteration", "y": "Reward"},
                               title=f"{name} Reward", )
    return dcc.Graph(figure=plot)


colors = ["red", "blue", "yellow", "green", "purple"]


def render_selection_indexes(products: List[Product], selected_price_indexes: List[PriceIndexes], name: str):
    x_iteration = list(range(1, len(selected_price_indexes) + 1))
    fig = go.Figure()
    for product in products:
        prices = []
        for selected_price_index in selected_price_indexes:
            prices.append(product.candidate_prices[selected_price_index[product.id]])
        fig.add_trace(go.Scatter(x=x_iteration, y=prices, name=product.name, line={"color": colors[product.id]}))
    fig.update_layout(title=f"{name} Prices", xaxis_title="Iteration", yaxis_title="Prices per product")
    return dcc.Graph(figure=fig)


def render_product_rewards_graph(products: List[Product], product_rewards: List[ProductRewards], name: str):
    x_iteration = list(range(1, len(product_rewards) + 1))
    fig = go.Figure()
    for product in products:
        fig.add_trace(
            go.Scatter(x=x_iteration, y=[product_reward[product.id] for product_reward in product_rewards],
                       name=product.name, line={"color": colors[product.id]}))

    fig.update_layout(title=f"{name} Product rewards", xaxis_title="Iteration", yaxis_title="Product Rewards")
    return dcc.Graph(figure=fig)


def render_for_learner(learner_name, learner_data):
    rewards = [reward for reward, _, _ in learner_data["exp"]]
    products = [Product(*product_dict.values()) for product_dict in learner_data["products"]]
    seleced_price_indexes = [index for _, index, _ in learner_data["exp"]]
    product_rewards = [p_reward for _, _, p_reward in learner_data["exp"]]

    return dbc.Col([
        dbc.Row(row) for row in [
            render_rewards(learner_name, rewards),
            render_selection_indexes(products, seleced_price_indexes, learner_name),
            render_product_rewards_graph(products, product_rewards, learner_name),
        ]
    ])


app.config.suppress_callback_exceptions = True

experiment_selector_left = dcc.Dropdown(
    id="experiment-selector-left",
)
experiment_selector_right = dcc.Dropdown(
    id="experiment-selector-right",
)


@app.callback(
    Output("experiment-selector-left", "options"),
    Output("experiment-selector-left", "value"),
    Output("experiment-selector-left", "disabled"),
    Output("experiment-selector-right", "options"),
    Output("experiment-selector-right", "value"),
    Output("experiment-selector-right", "disabled"),
    Input(data_storage, "data"),
)
def update_experiment_selectors(data):
    result_left = ([{"label": x, "value": x} for x in data["results"].keys()], list(data["results"].keys())[0], False) \
        if data is not None else ([], None, True)
    result_right = ([{"label": x, "value": x} for x in data["results"].keys()], list(data["results"].keys())[1], False) \
        if data is not None else ([], None, True)
    return *result_left, *result_right


@app.callback(
    Output("result-div-left", "children"),
    Input(data_storage, "data"),
    Input("experiment-selector-left", "value"),
)
def update_result_div_left(data, selected_experiment):
    if data is None:
        return html.Div()
    if selected_experiment is None:
        return html.Div("Pick an experiment from above to compare")
    return render_for_learner(selected_experiment, data["results"][selected_experiment])


@app.callback(
    Output("result-div-right", "children"),
    Input(data_storage, "data"),
    Input("experiment-selector-right", "value"),
)
def update_result_div_right(data, selected_experiment):
    if data is None:
        return html.Div()
    if selected_experiment is None:
        return html.Div("Pick an experiment from above to compare")
    return render_for_learner(selected_experiment, data["results"][selected_experiment])


results = dbc.Row([
    dbc.Col([
        dbc.Row([experiment_selector_left]),
        dbc.Row(id="result-div-left"),
    ], xs=6),
    dbc.Col([
        dbc.Row([experiment_selector_right]),
        dbc.Row(id="result-div-right"),

    ], xs=6)
], id="results")

app.layout = dbc.Container([
    data_storage,
    config_selectors,
    second_row,
    results,
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
