import time
from collections import namedtuple

import numpy as np
import plotly.express
import plotly.graph_objs as go
import scipy.ndimage
from dash import dcc, dash_table
import dash_bootstrap_components as dbc
from typing import List, Callable, Optional
from Learner import PriceIndexes, ProductRewards
from basic_types import Experience, Age
from entities import Product, Customer, reservation_price_distribution_from_curves
from change_detectors import ChangeHistoryItem
from web.common import SimulationResult


def apply_resolution(y, resolution):
    return scipy.ndimage.uniform_filter1d(y, size=resolution)


def render_rewards(name, rewards, change_detected_at: List[int], clairvoyant_rewards: List[float],
                   absolute_clairvoyant: float, resolution: int, std_dev):
    try:
        plot = plotly.express.line(x=range(1, len(rewards) + 1), y=apply_resolution(rewards, resolution),
                                   labels={"x": "Iteration", "y": "Reward"},
                                   title=f"{name} Reward", )
    except:
        time.sleep(0.1)
        plot = plotly.express.line(x=range(1, len(rewards) + 1), y=apply_resolution(rewards, resolution),
                                   labels={"x": "Iteration", "y": "Reward"},
                                   title=f"{name} Reward", )
    lower = rewards - std_dev
    upper = rewards + std_dev
    plot.add_trace(
        go.Scatter(x=list(range(1, len(rewards) + 1)), y=apply_resolution(lower, resolution), fill=None, mode='lines',
                   line={"color": "#000000"}, name="Lower bound"))
    plot.add_trace(
        go.Scatter(x=list(range(1, len(rewards) + 1)), y=apply_resolution(upper, resolution), fill=None, mode='lines',
                   line={"color": "#000000"}, name="Upper bound"))
    plot.add_trace(
        go.Scatter(x=list(range(1, len(rewards) + 1)), y=apply_resolution(clairvoyant_rewards, resolution), fill=None,
                   mode='lines',
                   line={"color": "darkblue"}, name="Clairvoyant reward"))
    for detected_i in change_detected_at:
        plot.add_vline(detected_i)
    plot.add_hline(absolute_clairvoyant, name="Absolute Clairvoyant reward")
    return dcc.Graph(figure=plot)


def render_regrets(name, regrets, change_detected_at: List[int], clairvoyant_rewards: List[float],
                   absolute_clairvoyant: float, resolution: int, std_dev):
    plot = plotly.express.line(x=range(1, len(regrets) + 1), y=apply_resolution(regrets, resolution),
                               labels={"x": "Iteration", "y": "Regret"},
                               title=f"{name} Regret", )
    lower = regrets - std_dev
    upper = regrets + std_dev
    plot.add_trace(
        go.Scatter(x=list(range(1, len(regrets) + 1)), y=apply_resolution(lower, resolution), fill=None, mode='lines',
                   line={"color": "#000000"}))
    plot.add_trace(
        go.Scatter(x=list(range(1, len(regrets) + 1)), y=apply_resolution(upper, resolution), fill=None, mode='lines',
                   line={"color": "#000000"}))
    plot.add_trace(
        go.Scatter(x=list(range(1, len(regrets) + 1)), y=apply_resolution(clairvoyant_rewards, resolution), fill=None,
                   mode='lines',
                   line={"color": "darkblue"}, name="Clairvoyant reward"))
    for detected_i in change_detected_at:
        plot.add_vline(detected_i)
    plot.add_hline(absolute_clairvoyant, name="Absolute Clairvoyant reward")
    return dcc.Graph(figure=plot)


def render_avg_regrets(name, avg_regrets, change_detected_at: List[int], clairvoyant_rewards: List[float],
                       absolute_clairvoyant: float, resolution: int):
    plot = plotly.express.line(x=range(1, len(avg_regrets) + 1), y=apply_resolution(avg_regrets, resolution),
                               labels={"x": "Iteration", "y": "Average Regret"},
                               title=f"{name} Average Regret", )
    plot.add_trace(
        go.Scatter(x=list(range(1, len(avg_regrets) + 1)), y=apply_resolution(clairvoyant_rewards, resolution),
                   fill=None,
                   mode='lines',
                   line={"color": "darkblue"}, name="Clairvoyant reward"))
    for detected_i in change_detected_at:
        plot.add_vline(detected_i)
    plot.add_hline(absolute_clairvoyant, name="Absolute Clairvoyant reward")
    return dcc.Graph(figure=plot)


colors = ["red", "blue", "yellow", "green", "purple", "orange", "pink", "brown", "olive", "cyan"]


def render_selection_indexes(products: List[Product], selected_price_indexes: List[PriceIndexes], name: str,
                             change_detected_at: List[int], resolution: int):
    x_iteration = list(range(1, len(selected_price_indexes) + 1))
    fig = go.Figure()
    from itertools import product as product_
    for product in products:
        if isinstance(selected_price_indexes[0], dict):
            prices = {(e, a): [] for e, a in product_(Experience, Age)}
            for price_indexes in selected_price_indexes:
                for k, v in price_indexes.items():
                    prices[k].append(product.candidate_prices[v[product.id]])
            for i, k in enumerate(price_indexes.keys(),start=1):
                fig.add_trace(
                    go.Scatter(x=x_iteration, y=apply_resolution(prices[k], resolution), name=f"{product.name} {k[0].value[:2]} {k[1].value[:2]}",
                               line={"color": colors[(product.id + i) % len(colors)]}))
        else:
            prices = []
            for selected_price_index in selected_price_indexes:
                prices.append(product.candidate_prices[selected_price_index[product.id]])
            fig.add_trace(go.Scatter(x=x_iteration, y=apply_resolution(prices, resolution), name=product.name,
                                 line={"color": colors[product.id]}))
    fig.update_layout(title=f"{name} Prices", xaxis_title="Iteration", yaxis_title="Prices per product")
    for detected_i in change_detected_at:
        fig.add_vline(detected_i)
    return dcc.Graph(figure=fig)


def render_product_rewards_graph(products: List[Product], product_rewards: List[ProductRewards], name: str,
                                 change_detected_at: List[int], resolution: int):
    x_iteration = list(range(1, len(product_rewards) + 1))
    fig = go.Figure()
    for product in products:
        fig.add_trace(
            go.Scatter(x=x_iteration,
                       y=apply_resolution([product_reward[product.id] for product_reward in product_rewards],
                                          resolution),
                       name=product.name, line={"color": colors[product.id]}))

    fig.update_layout(title=f"{name} Product rewards", xaxis_title="Iteration", yaxis_title="Product Rewards")
    for detected_i in change_detected_at:
        fig.add_vline(detected_i)
    return dcc.Graph(figure=fig)


def render_for_estimator(products: List[Product], type_history: List[List[float]], name: str, type_name: str):
    x_iteration = list(range(1, len(type_history[0]) + 1))
    fig = go.Figure()
    for product in products:
        fig.add_trace(
            go.Scatter(x=x_iteration, y=type_history[product.id],
                       name=product.name, line={"color": colors[product.id]}))

    fig.update_layout(title=f"{type_name} for estimator {name}", xaxis_title="Iteration",
                      yaxis_title=type_name)
    return dcc.Graph(figure=fig)


def render_change_detection_graph_graph(change_detection_history: List[ChangeHistoryItem],
                                        change_detected_at: List[int]):
    x = list(range(1, len(change_detection_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[item[0] for item in change_detection_history], name="gplus", mode='markers'))
    fig.add_trace(go.Scatter(x=x, y=[item[1] for item in change_detection_history], name="gminus", mode='markers'))
    fig.add_trace(go.Scatter(x=x, y=[item[2] for item in change_detection_history], name="sample", mode='markers'))
    for detected_i in change_detected_at:
        fig.add_vline(detected_i)
    return dcc.Graph(figure=fig)


def render_customer_table(customers: List[dict], products: List[Product], selected_price_indexes: PriceIndexes):
    data = []
    tooltips = []
    columns = {
        "name": "Name",
        "age": "Age",
        "class_": "Class",
        "entered_from": "First Product",
        "profit": "Profit",
        **{str(product.id): product.name for product in products},
    }
    columns_ = [{"name": v, "id": k} for k, v in columns.items()]
    from itertools import product as product_
    if isinstance(selected_price_indexes, list):
        selected_price_indexes = {
            (k, v): selected_price_indexes for k, v in product_(Experience, Age)
        }
    for customer in customers:
        data.append(dict(
            name=customer["display_name"],
            age=customer["display_age"],
            class_=str(customer["class"]),
            entered_from=products[customer["products_clicked"][0]].name if customer["products_clicked"] else "(out)",
            profit=sum(
                products[int(product_id)].candidate_prices[selected_price_indexes[(Experience[customer["expertise"]], Age[customer["age"]])][int(product_id)]] * count[0] for
                product_id, count in
                customer["products_bought"].items()),
            **{
                str(product.id): f"{'C ' if product.id in customer['products_clicked'] else ''}{('& B: ' + str(customer['products_bought'][str(product.id)][0]) if customer['products_bought'][str(product.id)][0] > 0 else '')}"
                for product in products},
        ))
        tooltips.append({product.id: dict(
            value=f"""The customer's reservation price was: *{customer['products_bought'][str(product.id)][1]}* from distribution *{reservation_price_distribution_from_curves(namedtuple("temp", ["name"])(customer['class']), product.id, product.candidate_prices[selected_price_indexes[(Experience[customer['expertise']], Age[customer['age']])][product.id]])}*.
Our offered price was: *{product.candidate_prices[selected_price_indexes[(Experience[customer["expertise"]], Age[customer["age"]])][product.id]]}*
""", type="markdown") for product in products})
    return dash_table.DataTable(
        columns=columns_,
        data=data,
        tooltip_data=tooltips,
        tooltip_delay=0,
        style_table={'overflowX': 'auto'},
        tooltip_duration=None,
        filter_action='native',
    )


def render_for_learner(learner_name: str, learner_data: List[SimulationResult], day_cnt: int, resolution: int = 10,
                       experiment_aggregator: Callable[[np.array], np.array] = None,
                       i_experiment: Optional[int] = None):
    rewards = experiment_aggregator(np.array([data.rewards for data in learner_data]))  # 1d float
    clairvoyant = experiment_aggregator(np.array([data.clairvoyants for data in learner_data]))  # 1d float
    absolute_clairvoyant = float(
        experiment_aggregator(np.array([data.absolute_clairvoyant for data in learner_data])))  # constant float
    change_detected_at = list(
        experiment_aggregator(np.array([data.change_detected_at for data in learner_data])).astype(
            int))  # 1d int
    product_rewards = experiment_aggregator(np.array([data.product_rewards for data in learner_data]))  # 2d float
    price_indexes = experiment_aggregator(np.array([data.price_indexes for data in learner_data]))  # 2d int
    products = learner_data[0].products
    regrets = np.cumsum(clairvoyant - rewards)
    # average_regrets = np.mean(np.sum(regrets, axis=0))
    sd_prof = np.std(rewards, axis=0) / np.sqrt(len(rewards))

    sd_reg = np.std(regrets, axis=0) / np.sqrt(len(rewards))

    graphs = [
        render_rewards(learner_name, np.array(rewards), change_detected_at, clairvoyant, absolute_clairvoyant,
                       resolution, sd_prof),
        render_regrets(learner_name, regrets, change_detected_at, clairvoyant, absolute_clairvoyant,
                       resolution, sd_reg),
        # render_avg_regrets(learner_name, average_regrets, change_detected_at, clairvoyant, absolute_clairvoyant,
        #                   resolution),
        render_selection_indexes(products, price_indexes, learner_name,
                                 change_detected_at, resolution),
        render_product_rewards_graph(products, product_rewards, learner_name,
                                     change_detected_at, resolution),
    ]
    if "Greedy" not in learner_name and i_experiment is not None:
        graphs.extend([
            dbc.FormText(f"Customers Day {day_cnt}:"),
            render_customer_table(learner_data[i_experiment].customers[day_cnt], products,
                                  learner_data[i_experiment].price_indexes[day_cnt])
        ])

    if learner_data[0].estimators is not None and learner_name.find("6") == -1:
        for name in list(learner_data[0].estimators[0].keys()):
            estimators = experiment_aggregator(
                np.array([[est[name] for est in data.estimators][1:] for data in learner_data]))

            graphs.extend(
                [render_for_estimator(products, estimators[:, 0].T, name, "Incoming factor"),
                 render_for_estimator(products, estimators[:, 1].T, name, "Resulting factor"),
                 render_for_estimator(products, estimators[:, 2].T, name, "Parameters factor"),

                 ])
    if i_experiment and learner_data[i_experiment].change_history:
        graphs.append(render_change_detection_graph_graph(learner_data[i_experiment].change_history,
                                                          learner_data[i_experiment].change_detected_at))
    return dbc.Col([
        dbc.Row(row) for row in graphs
    ])
