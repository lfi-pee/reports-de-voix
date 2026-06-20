from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import altair as alt
import polars as pl

from generation.donnees import NUANCE_VERS_BLOC, charger
from generation.graphiques import COULEUR

TEMPLATE = Path("docs/.analyse-template.html")
SORTIE = Path("docs/analyse-reports-legislatives-2024.html")
PARQUET = Path("generation/reports_legislatives_2024.parquet")
DATA = Path("/home/veesion/hexagonal/data/02_clean/elections")
CANDIDATS_T2 = DATA / "2024-legislatives-2-candidats.csv"
SEUIL_IC = 0.30
ABST = "non exprimés (T1)"


def _candidats() -> pl.DataFrame:
    """Candidat de second tour par (circonscription, bloc), pour enrichir les
    infobulles et les tableaux — issu du fichier officiel des candidatures."""
    return (
        pl.read_csv(CANDIDATS_T2, infer_schema_length=0)
        .with_columns(
            bloc=pl.col("nuance").replace_strict(NUANCE_VERS_BLOC, default="Divers"),
            candidat=pl.format("{} {}", pl.col("nom"), pl.col("prenom")),
        )
        .group_by("circonscription", "bloc")
        .agg(pl.col("candidat").first())
    )


def _avec_candidat(df: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    return df.join(
        lookup,
        left_on=["circonscription", "destination"],
        right_on=["circonscription", "bloc"],
        how="left",
    ).with_columns(pl.col("candidat").fill_null(""))


@dataclass(frozen=True)
class Graphe:
    div: str
    configuration: str | None  # None = toutes configurations
    paires: list[tuple[str, str]] = field(default_factory=list)


# Chaque graphe du rapport est régénéré à partir du parquet (report + IC bootstrap).
GRAPHES = [
    Graphe(
        "altair-viz-f2f19ba03acd4ecab652fb0666d2d1d4",
        "NFP / RN",
        [("ENS-HOR", "NFP"), ("LR", "NFP"), ("ENS-HOR", "RN"), ("LR", "RN")],
    ),
    Graphe(
        "altair-viz-605604cc8dee48f8b1fce47b3bce89dc",
        "ENS-HOR / RN",
        [("NFP", "ENS-HOR"), ("LR", "ENS-HOR"), ("NFP", "RN"), ("LR", "RN")],
    ),
    Graphe(
        "altair-viz-d0c42354d2df4c7d85dd4b39103e3706",
        "NFP / ENS-HOR / RN",
        [("LR", "NFP"), ("LR", "ENS-HOR"), ("LR", "RN")],
    ),
    Graphe(
        "altair-viz-5077ade42608437a8cbf81259402b12e",
        "NFP / ENS-HOR",
        [("LR", "NFP"), ("LR", "ENS-HOR"), ("RN", "NFP"), ("RN", "ENS-HOR")],
    ),
    Graphe(
        "altair-viz-6b8c5f4096bb4f8d865623dae2e3213f",
        "LR / RN",
        [("NFP", "LR"), ("NFP", "RN"), ("ENS-HOR", "LR"), ("ENS-HOR", "RN")],
    ),
    Graphe(
        "altair-viz-2268802d22a0495c976dba0fb5d6e2a0",
        None,
        [(ABST, "NFP"), (ABST, "ENS-HOR"), (ABST, "LR"), (ABST, "RN")],
    ),
]


def _extraire_spec(html: str, div: str) -> str:
    i = html.find(f'id="{div}"')
    j = html.find("})(", i)
    k = html.index("{", j + 3)
    depth = 0
    instr = esc = False
    for p in range(k, len(html)):
        ch = html[p]
        if instr:
            esc = ch == "\\" and not esc
            if ch == '"' and not esc:
                instr = False
        elif ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return html[k : p + 1]
    raise ValueError(f"spec introuvable : {div}")


TEXT_WIDTH = 800
GRIS_PT, GRIS_IC, GRIS_CLAIR = "#bcc1cc", "#8a93a3", "#dde0e6"
COLS = [
    "circonscription",
    "candidat",
    "report",
    "ic_bas",
    "ic_haut",
    "paire",
    "rang",
    "fiable",
    "coul",
    "coul_ic",
]


def _finaliser(df: pl.DataFrame, couleur: pl.Expr) -> pl.DataFrame:
    """Ajoute le rang (points régulièrement espacés par panneau, donc panneaux de même
    hauteur), la fiabilité (IC large = mal identifié, estompé) et les couleurs."""
    fiable = (pl.col("ic_haut") - pl.col("ic_bas")) <= SEUIL_IC
    return (
        df.with_columns(fiable=fiable)
        .with_columns(
            rang=(pl.col("report").rank("ordinal").over("paire") - 0.5)
            / pl.len().over("paire"),
            coul=pl.when("fiable").then(couleur).otherwise(pl.lit(GRIS_PT)),
            coul_ic=pl.when("fiable")
            .then(pl.lit(GRIS_IC))
            .otherwise(pl.lit(GRIS_CLAIR)),
        )
        .select(COLS)
    )


def _facette(
    data: pl.DataFrame, ordre: list[str], cols: int, cell_h: int = 600
) -> dict:
    largeur = (TEXT_WIDTH - 16 * (cols - 1)) / cols
    base = alt.Chart(data).encode(
        y=alt.Y("rang:Q", axis=None, scale=alt.Scale(domain=[0, 1]))
    )
    rule = base.mark_rule(strokeWidth=1).encode(
        x=alt.X("ic_bas:Q", scale=alt.Scale(domain=[0, 1]), title="report estimé"),
        x2="ic_haut:Q",
        color=alt.Color("coul_ic:N", scale=None, legend=None),
    )
    pt = base.mark_circle(size=34).encode(
        x=alt.X("report:Q", axis=alt.Axis(format="%")),
        color=alt.Color("coul:N", scale=None, legend=None),
        opacity=alt.condition("datum.fiable", alt.value(0.9), alt.value(0.45)),
    )
    # cible de survol transparente et large : points denses, zone sensible élargie.
    survol = base.mark_circle(size=260, opacity=0).encode(
        x=alt.X("report:Q"),
        tooltip=[
            alt.Tooltip("circonscription:N", title="Circ."),
            alt.Tooltip("candidat:N", title="Candidat·e"),
            alt.Tooltip("report:Q", title="Report", format=".0%"),
            alt.Tooltip("ic_bas:Q", title="IC bas", format=".0%"),
            alt.Tooltip("ic_haut:Q", title="IC haut", format=".0%"),
        ],
    )
    # molette/glissement pour zoomer et déplier les régions denses
    zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])
    panneau = (
        alt.layer(rule, pt, survol)
        .add_params(zoom)
        .properties(width=largeur, height=cell_h)
    )
    return (
        panneau.facet(
            facet=alt.Facet(
                "paire:N", title=None, sort=ordre, header=alt.Header(labelFontSize=12)
            ),
            columns=cols,
            spacing=16,
        )
        .configure_view(stroke=None)
        .to_dict()
    )


def _graphe(df: pl.DataFrame, cand: pl.DataFrame, g: Graphe) -> dict:
    base = (
        df
        if g.configuration is None
        else df.filter(pl.col("configuration") == g.configuration)
    )
    ordre = [f"{de} → {vers}" for de, vers in g.paires]
    filtre = base.with_columns(
        paire=pl.format("{} → {}", pl.col("source"), pl.col("destination"))
    ).filter(pl.col("paire").is_in(ordre))
    data = _finaliser(
        _avec_candidat(filtre, cand),
        pl.col("destination").replace_strict(COULEUR, default="#5a6172"),
    )
    return _facette(data, ordre, cols=3 if len(g.paires) == 3 else 2)


def _graphe_sensibilite(df: pl.DataFrame, cand: pl.DataFrame) -> dict:
    """Report des électeurs ENS-HOR et LR vers le candidat NFP, par sensibilité de
    celui-ci (PCF / FI / PS / EELV), reconstituée depuis inputs/sensibilites_nfp.csv."""
    sens = pl.read_csv("generation/inputs/sensibilites_nfp.csv")
    base = (
        df.filter(
            (pl.col("configuration") == "NFP / RN")
            & (pl.col("destination") == "NFP")
            & (pl.col("source").is_in(["ENS-HOR", "LR"]))
        )
        .join(sens, left_on="circonscription", right_on="circo", how="inner")
        .with_columns(
            paire=pl.format("{} · {}", pl.col("source"), pl.col("sensibilite"))
        )
    )
    data = _finaliser(_avec_candidat(base, cand), pl.lit(COULEUR["NFP"]))
    ordre = [
        f"{de} · {s}" for s in ["FI", "PS", "EELV", "PCF"] for de in ["ENS-HOR", "LR"]
    ]
    presents = set(data["paire"].unique().to_list())
    return _facette(data, [o for o in ordre if o in presents], cols=2, cell_h=258)


def _graphe_dissidents(df: pl.DataFrame, cand: pl.DataFrame) -> dict:
    """Report ENS-HOR / LR vers les deux candidats NFP dissidents opposés au RN
    (Ruffin 80-01, Davi 13-05) ; leur bloc de destination est la gauche."""
    base = df.filter(
        pl.col("circonscription").is_in(["13-05", "80-01"])
        & pl.col("source").is_in(["ENS-HOR", "LR"])
        & pl.col("destination").is_in(["NFP", "Gauche div."])
    ).with_columns(paire=pl.format("{} → NFP dissident", pl.col("source")))
    data = _finaliser(_avec_candidat(base, cand), pl.lit(COULEUR["NFP"]))
    ordre = ["ENS-HOR → NFP dissident", "LR → NFP dissident"]
    presents = set(data["paire"].unique().to_list())
    return _facette(data, [o for o in ordre if o in presents], cols=2, cell_h=144)


def _remplacer_tbody(html: str, marqueur: str, lignes: str) -> str:
    i = html.find(marqueur)
    a = html.find("<tbody>", i) + len("<tbody>")
    b = html.find("</tbody>", a)
    return html[:a] + lignes + html[b:]


def _table_r2_faible(df: pl.DataFrame) -> str:
    bas = (
        df.select("circonscription", "destination", "r2_echantillon")
        .unique()
        .filter(pl.col("r2_echantillon") < 0.90)
        .sort("r2_echantillon")
    )
    return "".join(
        f"<tr><td>{r['circonscription']}</td><td>{r['destination']}</td>"
        f"<td>{r['r2_echantillon']:.1%}</td></tr>"
        for r in bas.iter_rows(named=True)
    )


def _abstention_par_circo() -> pl.DataFrame:
    circos = charger(
        DATA / "2024-legislatives-1-bureau_de_vote.parquet",
        DATA / "2024-legislatives-2-bureau_de_vote.parquet",
        DATA / "2024-legislatives-correspondances-bureau_de_vote-circonscription.csv",
    )
    return pl.DataFrame(
        {
            "circonscription": [c.code for c in circos],
            "abstention": [float(c.t1[:, len(c.blocs_t1)].sum()) for c in circos],
        }
    )


def _table_mobilisation(df: pl.DataFrame, cand: pl.DataFrame) -> str:
    blocs = ["NFP", "ENS-HOR", "LR", "RN"]
    top = (
        df.filter((pl.col("source") == ABST) & pl.col("destination").is_in(blocs))
        .join(_abstention_par_circo(), on="circonscription", how="left")
        .join(
            cand,
            left_on=["circonscription", "destination"],
            right_on=["circonscription", "bloc"],
            how="left",
        )
        .with_columns(mobilises=(pl.col("report") * pl.col("abstention")).round())
        .sort("report", descending=True)
        .head(15)
    )
    return "".join(
        f"<tr><td>{r['circonscription']}</td><td>{r['destination']}</td>"
        f"<td>{r['candidat'] or ''}</td><td>{r['report']:.1%}</td>"
        f"<td>{int(r['mobilises'])}</td><td>{r['r2_echantillon']:.1%}</td></tr>"
        for r in top.iter_rows(named=True)
    )


def construire() -> None:
    html = TEMPLATE.read_text()
    df = pl.read_parquet(PARQUET)
    cand = _candidats()
    for g in GRAPHES:
        html = html.replace(
            _extraire_spec(html, g.div), json.dumps(_graphe(df, cand, g)), 1
        )
    speciaux = {
        "altair-viz-c3c8b54e784a4579a9ae63f2a8074374": _graphe_sensibilite,
        "altair-viz-dc3c3b0f29454558a7ce371c8ccbf3f5": _graphe_dissidents,
    }
    for div, builder in speciaux.items():
        html = html.replace(_extraire_spec(html, div), json.dumps(builder(df, cand)), 1)
    html = _remplacer_tbody(html, "inférieur à 90 %", _table_r2_faible(df))
    html = _remplacer_tbody(html, "plus hautes valeurs", _table_mobilisation(df, cand))
    SORTIE.write_text(html)
    print(f"régénéré {len(GRAPHES) + len(speciaux)} graphes + 2 tableaux -> {SORTIE}")


if __name__ == "__main__":
    construire()
