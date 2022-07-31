import dash

dash.register_page(__name__, order=1)
with open("web/pages/introduction.md", "r") as f:
    layout = dash.dcc.Markdown(f.read())
