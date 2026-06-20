from __future__ import annotations

from dataclasses import dataclass

import click
import numpy as np
import polars as pl
from tqdm import tqdm

from reports_de_voix.algorithme import (
    bootstrap_reports,
    calculer_r_square,
    calculer_r_square_validation,
    calculer_reports,
    conditionnement,
)


@dataclass
class Resultat:
    coefficients: np.ndarray
    r_square: np.ndarray
    r_square_cv: np.ndarray | None
    coefficients_std: np.ndarray | None
    conditionnement: float


def _preparer(resultats: pl.DataFrame) -> pl.DataFrame:
    return resultats.with_columns(
        bureau_de_vote=pl.format("{code_commune}-{bureau_de_vote}")
    )


def _bureaux(resultats: pl.DataFrame, key: str) -> pl.DataFrame:
    return (
        resultats.group_by("bureau_de_vote")
        .agg(
            pl.col(key).first(),
            pl.col("inscrits").first(),
            pl.col("voix").sum().alias("exprimés"),
        )
        .select(
            key,
            "bureau_de_vote",
            "inscrits",
            # non-exprimés = abstention + blancs + nuls (et non la seule abstention)
            (pl.col("inscrits") - pl.col("exprimés")).alias("non_exprimes"),
        )
    )


def _pivot(votes: pl.DataFrame, bureaux: pl.DataFrame) -> pl.DataFrame:
    return (
        votes.sort(["bureau_de_vote", "numero_panneau"])
        .pivot(
            on=["numero_panneau"],
            index=["bureau_de_vote"],
            values=["voix"],
            maintain_order=True,
        )
        .fill_null(0)
        .join(
            bureaux.select("bureau_de_vote", "non_exprimes", "inscrits"),
            on="bureau_de_vote",
            validate="1:1",
        )
    )


def _valeurs(pivot: pl.DataFrame) -> np.ndarray:
    return pivot.select(pl.all().exclude(["bureau_de_vote", "inscrits"])).to_numpy()


def _aligner(
    t1_pivot: pl.DataFrame, t2_pivot: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    communs = set(t1_pivot["bureau_de_vote"]) & set(t2_pivot["bureau_de_vote"])
    t1_pivot = t1_pivot.filter(pl.col("bureau_de_vote").is_in(communs)).sort(
        "bureau_de_vote"
    )
    t2_pivot = t2_pivot.filter(pl.col("bureau_de_vote").is_in(communs)).sort(
        "bureau_de_vote"
    )
    poids = 1.0 / np.sqrt(np.maximum(t2_pivot["inscrits"].to_numpy(), 1.0))
    return _valeurs(t1_pivot), _valeurs(t2_pivot), poids


def _traiter_unite(
    t1_votes: pl.DataFrame,
    t2_votes: pl.DataFrame,
    bureaux_t1: pl.DataFrame,
    bureaux_t2: pl.DataFrame,
    cv_splits: int,
    bootstrap: int,
) -> Resultat | None:
    t1_values, t2_values, poids = _aligner(
        _pivot(t1_votes, bureaux_t1), _pivot(t2_votes, bureaux_t2)
    )

    report = calculer_reports(t1_values, t2_values, poids=poids)
    if report is None:
        return None

    std = None
    if bootstrap:
        echantillon = bootstrap_reports(t1_values, t2_values, bootstrap, poids=poids)
        std = None if echantillon is None else echantillon[1].flatten("C")

    return Resultat(
        coefficients=report.flatten("C"),
        r_square=calculer_r_square(t1_values, t2_values, report),
        r_square_cv=calculer_r_square_validation(
            t1_values, t2_values, cv_splits, poids=poids
        )
        if cv_splits
        else None,
        coefficients_std=std,
        conditionnement=conditionnement(t1_values),
    )


@click.command()
@click.option("--t1", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--t2", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--key", required=True)
@click.option(
    "-o", "--output", type=click.Path(writable=True, dir_okay=False), required=True
)
@click.option(
    "--cv-splits",
    type=int,
    default=5,
    show_default=True,
    help="Plis pour le R² hors échantillon (0 pour le désactiver).",
)
@click.option(
    "--bootstrap",
    type=int,
    default=0,
    show_default=True,
    help="Rééchantillons pour l'écart type des coefficients (0 = désactivé).",
)
def main(
    t1: str, t2: str, key: str, output: str, cv_splits: int, bootstrap: int
) -> None:
    resultats_t1 = _preparer(pl.read_parquet(t1))
    resultats_t2 = _preparer(pl.read_parquet(t2))

    # une unité n'est identifiable que si elle compte au moins autant de bureaux que de
    # coefficients par colonne : nb_listes_t1 listes + non-exprimés + excédents
    comptes = resultats_t1.group_by(key).agg(
        pl.col("numero_panneau").n_unique().alias("nb_listes_t1"),
        pl.col("bureau_de_vote").n_unique().alias("nb_bureaux"),
    )
    unites = comptes.filter(pl.col("nb_bureaux") >= pl.col("nb_listes_t1") + 2)[
        key
    ].to_list()

    bureaux_t1 = _bureaux(resultats_t1, key).partition_by(key, as_dict=True)
    bureaux_t2 = _bureaux(resultats_t2, key).partition_by(key, as_dict=True)
    t1_par_unite = resultats_t1.partition_by(key, as_dict=True)
    t2_par_unite = resultats_t2.partition_by(key, as_dict=True)

    cles: list[object] = []
    resultats: list[Resultat] = []
    echecs = 0
    for unite in tqdm(unites):
        cle = (unite,)
        resultat = _traiter_unite(
            t1_par_unite[cle],
            t2_par_unite[cle],
            bureaux_t1[cle],
            bureaux_t2[cle],
            cv_splits,
            bootstrap,
        )
        if resultat is None:
            echecs += 1
            continue
        cles.append(unite)
        resultats.append(resultat)

    colonnes: dict[str, list] = {
        key: cles,
        "coefficients": [r.coefficients for r in resultats],
        "r_square": [r.r_square for r in resultats],
        "conditionnement": [r.conditionnement for r in resultats],
    }
    if cv_splits:
        colonnes["r_square_cv"] = [r.r_square_cv for r in resultats]
    if bootstrap:
        colonnes["coefficients_std"] = [r.coefficients_std for r in resultats]

    pl.DataFrame(colonnes).write_parquet(output)
    click.echo(
        f"{len(resultats)} traitées, {echecs} échecs sur {len(unites)} candidates."
    )


if __name__ == "__main__":
    main()
