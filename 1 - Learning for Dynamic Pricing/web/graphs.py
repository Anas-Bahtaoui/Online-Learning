import plotly.express
import plotly.graph_objs as go
import scipy.ndimage
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from typing import List
from Learner import PriceIndexes, ProductRewards
from entities import Product
from parameter_estimators import HistoryEntry
from web.common import SimulationResult


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


def render_for_estimator(products: List[Product], type_history: List[List[float]], name: str, type_name: str):
    x_iteration = list(range(1, len(type_history) + 1))
    fig = go.Figure()
    for product in products:
        fig.add_trace(
            go.Scatter(x=x_iteration, y=[history_item[product.id] for history_item in type_history],
                       name=product.name, line={"color": colors[product.id]}))

    fig.update_layout(title=f"{type_name} for estimator {name}", xaxis_title="Iteration",
                      yaxis_title=type_name)
    return dcc.Graph(figure=fig)


def render_for_learner(learner_name: str, learner_data: SimulationResult):
    graphs = [
        render_rewards(learner_name, learner_data.rewards),
        render_selection_indexes(learner_data.products, learner_data.price_indexes, learner_name),
        render_product_rewards_graph(learner_data.products, learner_data.product_rewards, learner_name),
    ]
    if learner_data.estimators is not None:
        for name, n_items_history in list(learner_data.estimators.items())[:2]:
            n_items_history = [HistoryEntry(*item) for item in n_items_history]
            incoming_list = [item.incoming_prices for item in n_items_history]
            result_list = [item.outgoing_prices for item in n_items_history]
            parameters = [item.parameter for item in n_items_history]
            graphs.extend(
                [render_for_estimator(learner_data.products, incoming_list, name, "Incoming factor"),
                 render_for_estimator(learner_data.products, result_list, name, "Resulting factor"),
                 render_for_estimator(learner_data.products, parameters, name, "Parameters factor"),

                 ])
    return dbc.Col([
        dbc.Row(row) for row in graphs
    ])
