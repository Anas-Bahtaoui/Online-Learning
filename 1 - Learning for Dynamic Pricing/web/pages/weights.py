from typing import List

import visdcc

from present_components import slide
from production import secondaries, product_configs
import dash
from dash import html
import dash_bootstrap_components as dbc


def secondary_to_graph(secondary: List[List[float]], node_id: str):
    return visdcc.Network(id=node_id,
                          options=dict(width="500px", height="500px"),
                          data=dict(
                              nodes=[{"id": id_, "label": config.name} for id_, config in enumerate(product_configs) if
                                     any(secondary[id_][to] > 0 for to in range(5)) or any(secondary[from_][id_] > 0 for from_ in range(5))],
                              edges=[
                                  {"id": f"{from_}-{to}", "from": from_, "to": to, "label": str(secondary[from_][to]),
                                   "width": secondary[from_][to], "arrows": "to"} for
                                  from_ in range(len(product_configs)) for to in range(len(product_configs)) if
                                  from_ != to and secondary[from_][to] > 0]
                          ))


cols = [
    [
        html.H3("Professional", className="card-title"),
        secondary_to_graph(secondaries.professional, "professional")
    ],
    [
        html.H3("Beginner Young", className="card-title"),
        secondary_to_graph(secondaries.young_beginner, "young_beginner")
    ],
    [
        html.H3("Beginner Old", className="card-title"),
        secondary_to_graph(secondaries.old_beginner, "old_beginner")
    ]
]
dash.register_page(__name__, order=2)
layout = slide("Weights of products", dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(content), className="border border-primary"), width=4, className="px-5") for content in cols
    ]),
], fluid=True, className="align-items-md-stretch mx-auto mt-3"))
