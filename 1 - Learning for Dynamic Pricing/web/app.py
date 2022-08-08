from dash import Dash, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from entities import SimulationConfig, Simulation
from production import LAMBDA_, product_configs, learners
from web.graphs import render_for_learner
from web.params import *
from common import ids, SimulationResult
from web.db_cache import *

from layout import layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


@app.callback(
    Output(ids.storage, "data"),
    Input(ids.run_experiment, "n_clicks"),
    Input(ids.reset_results, "n_clicks"),
    State(ids.storage, "data"),
    State(ids.run_count, "value"),
    [State(f"{i}-selector", "value") for i in range(4)],
)
def run_experiment(clicks, reset_clicks, saved_data, run_count, *config_values):
    learners_ = learners[7:]
    if saved_data["clicks"] == clicks:
        if saved_data["reset_clicks"] == reset_clicks:
            raise PreventUpdate
        saved_data["reset_clicks"] = reset_clicks
        blow_cache()
        if saved_data["results"] is not None:
            saved_data["results"] = None
        return saved_data
    saved_data["clicks"] = clicks
    config = SimulationConfig(
        dirichlets=potential_diriclets[config_values[0]],
        customer_counts=potential_customer_counts[config_values[1]],
        purchase_amounts=potential_purchase_amounts[config_values[2]],
        secondaries=potential_secondaries[config_values[3]],
        lambda_=LAMBDA_,
        product_configs=product_configs,
    )
    cache_key = (*(learner.name for learner in learners_), *config_values, run_count)
    db_result = get_cache(cache_key)
    if db_result is not None:
        output_data = db_result
    else:
        output_data = {}
        simulation = Simulation(config, learners_)
        simulation.run(run_count, log=False, plot_graphs=False, verbose=False)
        for learner in learners_:
            output_data[learner.name] = SimulationResult.from_result(learner, simulation).serialize()
        save_results(output_data, cache_key)
    saved_data["results"] = output_data
    return saved_data


# Result Graphs


app.config.suppress_callback_exceptions = True


@app.callback(
    Output(ids.left_experiment_selector, "options"),
    Output(ids.left_experiment_selector, "value"),
    Output(ids.left_experiment_selector, "disabled"),
    Output(ids.right_experiment_selector, "options"),
    Output(ids.right_experiment_selector, "value"),
    Output(ids.right_experiment_selector, "disabled"),
    Input(ids.storage, "data"),
)
def update_experiment_selectors(data):
    if data is None or data["results"] is None:
        result = ([], "(No learner data)", True)
        return *result, *result

    result_left = ([{"label": x, "value": x} for x in data["results"].keys()], list(data["results"].keys())[0], False)
    result_right = ([{"label": x, "value": x} for x in data["results"].keys()], list(data["results"].keys())[1], False)
    return *result_left, *result_right


@app.callback(
    Output(ids.result_div_left, "children"),
    Input(ids.storage, "data"),
    Input(ids.left_experiment_selector, "value"),
)
def update_result_div_left(data, selected_experiment):
    if data is None or data["results"] is None:
        return html.Div("Run a simulation to compare results")
    if selected_experiment is None:
        return html.Div("Pick an experiment from above to compare")
    results = SimulationResult.deserialize(data["results"][selected_experiment])
    return render_for_learner(selected_experiment, results)


@app.callback(
    Output(ids.result_div_right, "children"),
    Input(ids.storage, "data"),
    Input(ids.right_experiment_selector, "value"),
)
def update_result_div_right(data, selected_experiment):
    if data is None or data["results"] is None:
        return html.Div()
    if selected_experiment is None:
        return html.Div("Pick an experiment from above to compare")
    results = SimulationResult.deserialize(data["results"][selected_experiment])
    return render_for_learner(selected_experiment, results)


app.layout = layout
