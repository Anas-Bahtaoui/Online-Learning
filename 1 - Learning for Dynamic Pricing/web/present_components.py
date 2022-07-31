from dash import html


def slide(title: str, body=None):
    if body is None:
        return html.Div([
            html.H1(title),
        ], style={"text-align": "center", "margin": "auto"})
    return html.Div([
        html.H4(title, style={"text-align": "center", "margin": "0 auto", "padding": "20px"}),
        body,
    ])
