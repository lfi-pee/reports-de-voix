from __future__ import annotations

import altair as alt
import polars as pl

alt.data_transformers.enable("default", max_rows=100_000)

ABST = "non exprimés (T1)"
TEXT_WIDTH = 800
GRIS_STAB = "#8a93a3"

COULEUR = {
    "NFP": "#e8443a",
    "ENS-HOR": "#f2a900",
    "LR": "#0a6cb0",
    "RN": "#1d3b6e",
    "Reconquête": "#5b2a86",
    "Union ext. droite": "#3a4a7a",
}

COLS = [
    "circonscription",
    "candidat",
    "report",
    "stabilite_bas",
    "stabilite_haut",
    "paire",
    "rang",
    "coul",
]


def graphe_configurations(conf: pl.DataFrame) -> alt.LayerChart:
    """Anneau des configurations de second tour (colonnes ``configuration`` et
    ``nombre``, telles que produites par ``tables.configurations``)."""
    base = alt.Chart(conf.with_row_index("ordre")).encode(
        theta=alt.Theta("nombre:Q", stack=True, sort=alt.SortField("ordre")),
        order=alt.Order("ordre:Q"),
        color=alt.Color("configuration:N", legend=None),
    )
    anneau = base.mark_arc(innerRadius=100, outerRadius=140)
    labels = base.mark_text(radius=180, size=10, fontWeight="bold").encode(
        text="configuration:N"
    )
    return (anneau + labels).properties(width=500, height=500)


def _avec_candidat(df: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    return df.join(
        lookup,
        left_on=["circonscription", "destination"],
        right_on=["circonscription", "bloc"],
        how="left",
    ).with_columns(pl.col("candidat").fill_null(""))


def _finaliser(df: pl.DataFrame, couleur: pl.Expr) -> pl.DataFrame:
    """Ajoute le rang (points régulièrement espacés par panneau, donc panneaux de même
    hauteur) et la couleur du point."""
    return df.with_columns(
        rang=(pl.col("report").rank("ordinal").over("paire") - 0.5)
        / pl.len().over("paire"),
        coul=couleur,
    ).select(COLS)


def _panneau(data: pl.DataFrame, label: str, largeur: int, cell_h: int, idx: int):
    sub = data.filter(pl.col("paire") == label)
    base = sub.pipe(alt.Chart).encode(
        y=alt.Y("rang:Q", axis=None, scale=alt.Scale(domain=[0, 1]))
    )
    rule = base.mark_rule(strokeWidth=1, color=GRIS_STAB).encode(
        x=alt.X(
            "stabilite_bas:Q", scale=alt.Scale(domain=[0, 1]), title="report estimé"
        ),
        x2="stabilite_haut:Q",
    )
    pt = base.mark_circle(size=34, opacity=0.9).encode(
        x=alt.X("report:Q", axis=alt.Axis(format="%")),
        color=alt.Color("coul:N", scale=None, legend=None),
    )
    # cible de survol transparente et large : points denses, zone sensible élargie.
    survol = base.mark_circle(size=260, opacity=0).encode(
        x=alt.X("report:Q"),
        tooltip=[
            alt.Tooltip("circonscription:N", title="Circ."),
            alt.Tooltip("candidat:N", title="Candidat·e"),
            alt.Tooltip("report:Q", title="Report", format=".0%"),
            alt.Tooltip("stabilite_bas:Q", title="Plage basse", format=".0%"),
            alt.Tooltip("stabilite_haut:Q", title="Plage haute", format=".0%"),
        ],
    )
    # un zoom propre à chaque panneau (molette/glissement), indépendant des autres
    zoom = alt.selection_interval(
        bind="scales", encodings=["x", "y"], name=f"zoom{idx}"
    )
    return (
        alt.layer(rule, pt, survol)
        .add_params(zoom)
        .properties(
            width=largeur, height=cell_h, title=alt.TitleParams(label, fontSize=12)
        )
    )


def _facette(
    data: pl.DataFrame, ordre: list[str], cols: int, cell_h: int = 600
) -> alt.VConcatChart:
    largeur = (TEXT_WIDTH - 16 * (cols - 1)) / cols
    presents = set(data["paire"].unique().to_list())
    labels = [lab for lab in ordre if lab in presents]
    panneaux = [_panneau(data, lab, largeur, cell_h, i) for i, lab in enumerate(labels)]
    lignes = [
        alt.hconcat(*panneaux[i : i + cols]) for i in range(0, len(panneaux), cols)
    ]
    return alt.vconcat(*lignes, spacing=24).configure_view(stroke=None)


def paires(df: pl.DataFrame) -> pl.DataFrame:
    """Ajoute la colonne ``paire`` (« source → destination »)."""
    return df.with_columns(
        paire=pl.format("{} → {}", pl.col("source"), pl.col("destination"))
    )


def panneaux(
    df: pl.DataFrame,
    cand: pl.DataFrame,
    ordre: list[str],
    cols: int = 2,
    cell_h: int = 600,
    couleur: str | None = None,
) -> alt.VConcatChart:
    """Un panneau de reports par valeur de ``ordre`` (colonne ``paire``) : un point
    par circonscription, plage de stabilité bootstrap en fond. Points colorés selon
    le bloc de destination, sauf ``couleur`` (nom d'un bloc) imposée."""
    expr = (
        pl.col("destination").replace_strict(COULEUR, default="#5a6172")
        if couleur is None
        else pl.lit(COULEUR[couleur])
    )
    sel = df.filter(pl.col("paire").is_in(ordre))
    data = _finaliser(_avec_candidat(sel, cand), expr)
    return _facette(data, ordre, cols=cols, cell_h=cell_h)
