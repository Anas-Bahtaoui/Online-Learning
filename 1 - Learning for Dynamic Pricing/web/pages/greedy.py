import os.path

import dash
from dash import html
from carbon_now import generate_from_code
from present_components import slide

dash.register_page(__name__, order=3)
code_path = os.path.join(os.path.dirname(__file__), "..", "..", "Clairvoyant.py")
layout = slide("Greedy Code",
               generate_from_code(code_path))
