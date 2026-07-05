"""Convertit rapport.ipynb en site/index.html : les graphes Vega-Lite embarqués
dans le notebook (mime application/vnd.vegalite.v6+json, ignoré par nbconvert au
profit du PNG statique) sont remplacés par des vues vega-embed interactives."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

VEGA_MIME = "application/vnd.vegalite.v6+json"
# nbconvert embarque require.js : define.amd étant présent, les bundles UMD de vega
# s'enregistrent en modules AMD au lieu de poser window.vegaEmbed. On masque define
# le temps de leur chargement.
CDN = (
    "<script>var defineAmd = window.define; window.define = undefined;</script>"
    '<script src="https://cdn.jsdelivr.net/npm/vega@6"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>'
    "<script>window.define = defineAmd;</script>"
)


def main() -> None:
    nb = nbformat.read("rapport.ipynb", as_version=4)
    n = 0
    for cell in nb.cells:
        for sortie in cell.get("outputs", []):
            spec = sortie.get("data", {}).get(VEGA_MIME)
            if spec is None:
                continue
            n += 1
            sortie["data"] = {
                "text/html": (
                    f'<div id="graphe{n}"></div>\n<script>'
                    f'vegaEmbed("#graphe{n}", {json.dumps(spec)}, '
                    "{actions: false});</script>"
                )
            }
    html, _ = HTMLExporter(theme="dark").from_notebook_node(nb)
    html = html.replace("</head>", CDN + "</head>", 1)
    sortie_html = Path("site/index.html")
    sortie_html.parent.mkdir(exist_ok=True)
    sortie_html.write_text(html)
    print(f"{n} graphes interactifs -> {sortie_html}")


if __name__ == "__main__":
    main()
