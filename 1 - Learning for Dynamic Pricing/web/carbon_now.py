from typing import Optional, Tuple

from dash import html
from urllib.parse import quote


def generate_from_code(code_path: str, between_lines: Optional[Tuple[int, int]] = None):
    """
    Generate a carbon now image from a code.
    """
    with open(code_path, "r") as file:
        content = file.read()
        if between_lines is not None:
            content = "\n".join(content.split("\n")[between_lines[0]:between_lines[1]])
    url_encoded_content = quote(content)
    height_px = max(len(content.split("\n")) * 20, 500)

    return html.Iframe(
        src=f"https://carbon.now.sh/embed?bg=rgba%28255%2C255%2C255%2C1%29&t=vscode&wt=none&l=python&width=680&ds=false&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=false&fl=1&fm=Fira+Code&fs=14px&lh=152%25&si=false&es=2x&wm=false&code={url_encoded_content}",
        style={"width": "100%", "height": f"{height_px}px", "border": "0", "transform": "scale(1)", "overflow": "hidden"},
        sandbox="allow-scripts allow-same-origin",
    )
