from __future__ import annotations

import altair as alt
import polars as pl

alt.data_transformers.enable("default", max_rows=100_000)

COULEUR = {
    "NFP": "#e8443a",
    "ENS-HOR": "#f2a900",
    "LR": "#0a6cb0",
    "RN": "#1d3b6e",
    "Reconquête": "#5b2a86",
    "Union ext. droite": "#3a4a7a",
}


def reports_tries(
    df: pl.DataFrame,
    configuration: str,
    source: str,
    destination: str,
    titre: str | None = None,
    seuil_ic: float = 0.30,
) -> alt.LayerChart:
    """Reports d'un bloc source vers un bloc destination, une ligne par
    circonscription, triées du plus faible au plus fort report, avec intervalle de
    confiance bootstrap à 95 %. Les coefficients dont l'intervalle dépasse ``seuil_ic``
    sont jugés mal identifiés et estompés : c'est leur intervalle, non leur point, qui
    fait foi."""

    sel = (
        df.filter(
            (pl.col("configuration") == configuration)
            & (pl.col("source") == source)
            & (pl.col("destination") == destination)
        )
        .with_columns(
            (pl.col("report") * 100).alias("report_pct"),
            ((pl.col("ic_haut") - pl.col("ic_bas")) <= seuil_ic).alias("fiable"),
        )
        .sort("report")
    )
    data = sel
    ordre = sel["circonscription"].to_list()
    couleur = COULEUR.get(destination, "#444")

    base = alt.Chart(data).encode(
        y=alt.Y(
            "circonscription:N",
            sort=ordre,
            title=None,
            axis=alt.Axis(labelFontSize=9, ticks=False, domain=False),
        )
    )
    barre = base.mark_rule(strokeWidth=1.4).encode(
        x=alt.X("ic_bas:Q", scale=alt.Scale(domain=[0, 1]), title="Report estimé"),
        x2="ic_haut:Q",
        color=alt.condition("datum.fiable", alt.value("#8a93a3"), alt.value("#d6dae1")),
    )
    point = base.mark_circle(size=42).encode(
        x=alt.X("report:Q", axis=alt.Axis(format="%")),
        color=alt.condition("datum.fiable", alt.value(couleur), alt.value("#b9bec9")),
        opacity=alt.condition("datum.fiable", alt.value(0.95), alt.value(0.45)),
        tooltip=[
            alt.Tooltip("circonscription:N", title="Circ."),
            alt.Tooltip("report_pct:Q", title="Report", format=".0f"),
            alt.Tooltip("ic_bas:Q", title="IC bas", format=".0%"),
            alt.Tooltip("ic_haut:Q", title="IC haut", format=".0%"),
            alt.Tooltip("n_bureaux:Q", title="Bureaux"),
        ],
    )
    n_fiable = int(sel["fiable"].sum())
    titre = titre or f"{source} → {destination} ({configuration})"
    sous = f"{len(ordre)} circonscriptions, dont {n_fiable} estimées précisément"
    return (barre + point).properties(
        height=alt.Step(13),
        width=460,
        title=alt.TitleParams(titre, subtitle=sous),
    )


def synthese_distribution(
    df: pl.DataFrame, configuration: str, source: str, destination: str
) -> alt.Chart:
    """Distribution des reports (un point par circonscription) avec médiane, pour
    juger de la concentration des valeurs."""
    sel = df.filter(
        (pl.col("configuration") == configuration)
        & (pl.col("source") == source)
        & (pl.col("destination") == destination)
    )
    return (
        alt.Chart(sel)
        .mark_tick(thickness=2, color=COULEUR.get(destination, "#444"))
        .encode(
            x=alt.X(
                "report:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")
            ),
            tooltip=["circonscription:N", alt.Tooltip("report:Q", format=".0%")],
        )
        .properties(width=460, height=60, title=f"{source} → {destination}")
    )
