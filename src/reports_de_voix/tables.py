from __future__ import annotations

import polars as pl

from reports_de_voix.donnees import DATA, NUANCE_VERS_BLOC, charger
from reports_de_voix.graphiques import ABST

CIRCO_EXEMPLE = "80-02"  # 2e circonscription de la Somme, citée en exemple
COLS_EXEMPLE = ["NFP", "RN", "ENS-HOR"]


def configurations(df: pl.DataFrame, garder: int = 8) -> pl.DataFrame:
    """Effectif et part de chaque configuration de second tour, les configurations
    au-delà des ``garder`` plus fréquentes regroupées sous « Autres »."""
    conf = (
        df.unique(subset=["circonscription"])
        .group_by("configuration")
        .agg(nombre=pl.len())
        .sort("nombre", descending=True)
        .with_columns(
            configuration=pl.when(pl.int_range(pl.len()) < garder)
            .then("configuration")
            .otherwise(pl.lit("Autres"))
        )
        .group_by("configuration", maintain_order=True)
        .agg(pl.col("nombre").sum())
    )
    return conf.with_columns(part=pl.col("nombre") / pl.col("nombre").sum())


def matrice_exemple(df: pl.DataFrame) -> pl.DataFrame:
    """Matrice de report de la circonscription donnée en exemple (en %), une ligne
    par source de premier tour."""
    ordre = [*dict.fromkeys(NUANCE_VERS_BLOC.values()), ABST]
    piv = df.filter(pl.col("circonscription") == CIRCO_EXEMPLE).pivot(
        on="destination", index="source", values="report"
    )
    presents = [s for s in ordre if s in set(piv["source"].to_list())]
    return (
        pl.DataFrame({"source": presents})
        .join(piv, on="source", how="left")
        .select(
            pl.col("source").replace({ABST: "abstention"}).alias("T1 \\ T2"),
            *(
                ((pl.col(c) if c in piv.columns else pl.lit(0.0)) * 100)
                .round(1)
                .alias(c)
                for c in COLS_EXEMPLE
            ),
        )
    )


def r2_faible(df: pl.DataFrame) -> pl.DataFrame:
    """Couples (circonscription, destination) dont le R² en échantillon est
    inférieur à 90 %."""
    return (
        df.select("circonscription", "destination", "r2_echantillon")
        .unique()
        .filter(pl.col("r2_echantillon") < 0.90)
        .sort("r2_echantillon")
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


def mobilisation(df: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """Les 15 plus hauts taux de report depuis les non-exprimés du premier tour,
    avec le volume d'abstentionnistes mobilisés correspondant."""
    blocs = ["NFP", "ENS-HOR", "LR", "RN"]
    return (
        df.filter((pl.col("source") == ABST) & pl.col("destination").is_in(blocs))
        .join(_abstention_par_circo(), on="circonscription", how="left")
        .join(
            cand,
            left_on=["circonscription", "destination"],
            right_on=["circonscription", "bloc"],
            how="left",
        )
        .with_columns(mobilises=(pl.col("report") * pl.col("abstention")).round(0))
        .sort("report", descending=True)
        .head(15)
        .select(
            "circonscription",
            "destination",
            "candidat",
            "report",
            "mobilises",
            "r2_echantillon",
        )
    )
