import plotly.express
import plotly.graph_objs as go
import scipy.ndimage
from dash import dcc, dash_table
import dash_bootstrap_components as dbc
from typing import List
from Learner import PriceIndexes, ProductRewards
from entities import Product, Customer
from parameter_estimators import HistoryEntry
from change_detectors import ChangeHistoryItem
from web.common import SimulationResult


def render_rewards(name, rewards, change_detected_at: List[int], clairvoyant_reward: float):
    plot = plotly.express.line(x=range(1, len(rewards) + 1), y=scipy.ndimage.uniform_filter1d(rewards, size=10),
                               labels={"x": "Iteration", "y": "Reward"},
                               title=f"{name} Reward", )
    for detected_i in change_detected_at:
        plot.add_vline(detected_i)
    plot.add_hline(clairvoyant_reward, name="Clairvoyant reward")
    return dcc.Graph(figure=plot)


colors = ["red", "blue", "yellow", "green", "purple"]


def render_selection_indexes(products: List[Product], selected_price_indexes: List[PriceIndexes], name: str,
                             change_detected_at: List[int]):
    x_iteration = list(range(1, len(selected_price_indexes) + 1))
    fig = go.Figure()
    for product in products:
        prices = []
        for selected_price_index in selected_price_indexes:
            prices.append(product.candidate_prices[selected_price_index[product.id]])
        fig.add_trace(go.Scatter(x=x_iteration, y=prices, name=product.name, line={"color": colors[product.id]}))
    fig.update_layout(title=f"{name} Prices", xaxis_title="Iteration", yaxis_title="Prices per product")
    for detected_i in change_detected_at:
        fig.add_vline(detected_i)
    return dcc.Graph(figure=fig)


def render_product_rewards_graph(products: List[Product], product_rewards: List[ProductRewards], name: str,
                                 change_detected_at: List[int]):
    x_iteration = list(range(1, len(product_rewards) + 1))
    fig = go.Figure()
    for product in products:
        fig.add_trace(
            go.Scatter(x=x_iteration, y=[product_reward[product.id] for product_reward in product_rewards],
                       name=product.name, line={"color": colors[product.id]}))

    fig.update_layout(title=f"{name} Product rewards", xaxis_title="Iteration", yaxis_title="Product Rewards")
    for detected_i in change_detected_at:
        fig.add_vline(detected_i)
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


def render_customer_table(customers: List[Customer], products: List[Product], selected_price_index: PriceIndexes):
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
    for customer in customers:
        data.append(dict(
            name=customer.display_name,
            age=customer.display_age,
            class_=str(customer.class_),
            entered_from=products[customer.products_clicked[0]].name if customer.products_clicked else "(out)",
            profit=sum(
                products[int(product_id)].candidate_prices[selected_price_index[int(product_id)]] * count for product_id, count in
                customer.products_bought.items()),
            **{
                str(product.id): f"{'C ' if product.id in customer.products_clicked else ''}{('& B: ' + str(customer.products_bought[str(product.id)]) if customer.products_bought[str(product.id)] > 0 else '')}"
                for product in products},
        ))
        tooltips.append({product.id: dict(
            value=f"""The customer's reservation price was: *{customer.get_reservation_price_of(product.id, product.candidate_prices[selected_price_index[product.id]])}*
Our offered price was: *{product.candidate_prices[selected_price_index[product.id]]}*
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

def render_for_learner(learner_name: str, learner_data: SimulationResult):
    graphs = [
        render_rewards(learner_name, learner_data.rewards, learner_data.change_detected_at, learner_data.clairvoyant),
        render_selection_indexes(learner_data.products, learner_data.price_indexes, learner_name,
                                 learner_data.change_detected_at),
        render_product_rewards_graph(learner_data.products, learner_data.product_rewards, learner_name,
                                     learner_data.change_detected_at),
        dbc.FormText("Customers Day 0:"),
        # render_customer_table(learner_data.customers[0], learner_data.products, learner_data.price_indexes[0])
    ]
    if learner_data.estimators is not None and learner_name.find("6") == -1:
        for name, n_items_history in list(learner_data.estimators.items()):  # [:2]:
            n_items_history = [HistoryEntry(*item) for item in n_items_history]
            incoming_list = [item.incoming_prices for item in n_items_history]
            result_list = [item.outgoing_prices for item in n_items_history]
            parameters = [item.parameter for item in n_items_history]
            graphs.extend(
                [render_for_estimator(learner_data.products, incoming_list, name, "Incoming factor"),
                 render_for_estimator(learner_data.products, result_list, name, "Resulting factor"),
                 render_for_estimator(learner_data.products, parameters, name, "Parameters factor"),

                 ])
    if learner_data.change_history:
        graphs.append(render_change_detection_graph_graph(learner_data.change_history, learner_data.change_detected_at))
    return dbc.Col([
        dbc.Row(row) for row in graphs
    ])
