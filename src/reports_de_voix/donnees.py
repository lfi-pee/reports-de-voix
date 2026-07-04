from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

RACINE = Path(__file__).resolve().parents[2]
PARQUET = RACINE / "data" / "reports_legislatives_2024.parquet"
SENSIBILITES = RACINE / "data" / "sensibilites_nfp.csv"
DATA = Path("/home/veesion/hexagonal/data/02_clean/elections")
CANDIDATS_T2 = DATA / "2024-legislatives-2-candidats.csv"

# Regroupement des nuances officielles en blocs interprétables. Le nuançage candidat
# fin (sensibilités internes au NFP, dissidences) n'est pas disponible dans les seules
# nuances ministérielles : ces blocs sont le niveau d'analyse le plus fin reproductible.
NUANCE_VERS_BLOC: dict[str, str] = {
    "UG": "NFP",
    "FI": "NFP",
    "SOC": "NFP",
    "VEC": "NFP",
    "ECO": "NFP",
    "COM": "NFP",
    "RDG": "NFP",
    "DVG": "Gauche div.",
    "EXG": "Gauche div.",
    "ENS": "ENS-HOR",
    "HOR": "ENS-HOR",
    "UDI": "Centre-droit",
    "DVC": "Centre-droit",
    "LR": "LR",
    "DVD": "Droite div.",
    "REC": "Reconquête",
    "UXD": "Union ext. droite",
    "EXD": "Union ext. droite",
    "DSV": "Union ext. droite",
    "RN": "RN",
    "REG": "Régionaliste",
    "DIV": "Divers",
}


@dataclass(frozen=True)
class Circonscription:
    code: str
    blocs_t1: list[str]
    blocs_t2: list[str]
    t1: np.ndarray  # bureaux × (blocs_t1 + non_exprimes)
    t2: np.ndarray  # bureaux × (blocs_t2 + non_exprimes)
    poids: np.ndarray  # 1/√inscrits par bureau


def _normaliser(resultats: pl.DataFrame) -> pl.DataFrame:
    # T2 préfixe les bureaux de zéros ("0001"), la correspondance et T1 non ("1") ;
    # on aligne en retirant les zéros de tête (en gardant les suffixes alpha, "23B").
    return resultats.with_columns(
        pl.col("bureau_de_vote").str.strip_chars_start("0"),
        pl.col("nuance")
        .replace_strict(NUANCE_VERS_BLOC, default="Divers")
        .alias("bloc"),
    )


def _table_blocs(resultats: pl.DataFrame, correspondance: pl.DataFrame) -> pl.DataFrame:
    inscrits = resultats.group_by("code_commune", "bureau_de_vote").agg(
        pl.col("inscrits").first()
    )
    par_bloc = (
        resultats.group_by("code_commune", "bureau_de_vote", "bloc")
        .agg(pl.col("voix").sum())
        .join(correspondance, on=["code_commune", "bureau_de_vote"])
        .join(inscrits, on=["code_commune", "bureau_de_vote"])
    )
    exprimes = par_bloc.group_by("code_commune", "bureau_de_vote").agg(
        pl.col("voix").sum().alias("exprimes"), pl.col("inscrits").first()
    )
    pivot = (
        par_bloc.pivot(
            on="bloc",
            index=["circonscription", "code_commune", "bureau_de_vote"],
            values="voix",
            aggregate_function="sum",
        )
        .fill_null(0)
        .join(exprimes, on=["code_commune", "bureau_de_vote"])
        .with_columns(
            non_exprimes=(pl.col("inscrits") - pl.col("exprimes")).clip(lower_bound=0),
            bureau=pl.format("{}-{}", pl.col("code_commune"), pl.col("bureau_de_vote")),
        )
    )
    return pivot


def _matrice(pivot: pl.DataFrame, blocs: list[str]) -> np.ndarray:
    return pivot.select([*blocs, "non_exprimes"]).to_numpy().astype(float)


def candidats() -> pl.DataFrame:
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


def charger(
    t1_path: Path, t2_path: Path, correspondance_path: Path
) -> list[Circonscription]:
    """Résultats T1/T2 par bureau, agrégés en blocs et alignés sur les bureaux
    présents aux deux tours, une matrice par circonscription."""
    correspondance = pl.read_csv(
        correspondance_path, infer_schema_length=0
    ).with_columns(pl.col("bureau_de_vote").str.strip_chars_start("0"))
    t1 = _table_blocs(_normaliser(pl.read_parquet(t1_path)), correspondance)
    t2 = _table_blocs(_normaliser(pl.read_parquet(t2_path)), correspondance)

    circos: list[Circonscription] = []
    for code in sorted(set(t1["circonscription"]) & set(t2["circonscription"])):
        p1 = t1.filter(pl.col("circonscription") == code)
        p2 = t2.filter(pl.col("circonscription") == code)
        communs = sorted(set(p1["bureau"]) & set(p2["bureau"]))
        if not communs:
            continue
        p1 = p1.filter(pl.col("bureau").is_in(communs)).sort("bureau")
        p2 = p2.filter(pl.col("bureau").is_in(communs)).sort("bureau")
        ordre = list(dict.fromkeys(NUANCE_VERS_BLOC.values()))
        # un bloc sans voix dans la circonscription n'est ni une source ni une
        # destination : sa colonne, nulle, rendrait la matrice de rang déficient.
        blocs_t1 = [b for b in ordre if b in p1.columns and p1[b].sum() > 0]
        blocs_t2 = [b for b in ordre if b in p2.columns and p2[b].sum() > 0]
        inscrits = p2["inscrits"].to_numpy().astype(float)
        circos.append(
            Circonscription(
                code=code,
                blocs_t1=blocs_t1,
                blocs_t2=blocs_t2,
                t1=_matrice(p1, blocs_t1),
                t2=_matrice(p2, blocs_t2),
                poids=1.0 / np.sqrt(np.maximum(inscrits, 1.0)),
            )
        )
    return circos
