from __future__ import annotations

import click
import numpy as np
import polars as pl
from tqdm import tqdm

from reports_de_voix.algorithme import (
    bootstrap_reports,
    calculer_r_square,
    calculer_r_square_validation,
    calculer_reports,
)


def _preparer(resultats: pl.DataFrame) -> pl.DataFrame:
    return resultats.with_columns(
        bureau_de_vote=pl.format("{code_commune}-{bureau_de_vote}")
    )


def _abstention(resultats: pl.DataFrame, key: str) -> pl.DataFrame:
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
            (pl.col("inscrits") - pl.col("exprimés")).alias("abstention"),
        )
    )


def _matrice(votes: pl.DataFrame, abstention: pl.DataFrame) -> np.ndarray:
    pivot = (
        votes.sort(["bureau_de_vote", "numero_panneau"])
        .pivot(
            on=["numero_panneau"],
            index=["bureau_de_vote"],
            values=["voix"],
            maintain_order=True,
        )
        .join(
            abstention.select("bureau_de_vote", "abstention"),
            on=["bureau_de_vote"],
            validate="1:1",
            maintain_order="left",
        )
    )
    return pivot.select(pl.all().exclude(["bureau_de_vote"])).to_numpy()


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
    help="Nombre de plis pour le R² hors échantillon (0 pour le désactiver).",
)
@click.option(
    "--bootstrap",
    type=int,
    default=0,
    show_default=True,
    help="Nombre de rééchantillons pour l'écart type des coefficients (0 = désactivé).",
)
def main(
    t1: str, t2: str, key: str, output: str, cv_splits: int, bootstrap: int
) -> None:
    resultats_t1 = _preparer(pl.read_parquet(t1))
    resultats_t2 = _preparer(pl.read_parquet(t2))

    # une unité n'est identifiable que si elle compte au moins autant de bureaux que de
    # coefficients à estimer par colonne : nb_listes_t1 listes + abstention + excédents
    comptes = resultats_t1.group_by(key).agg(
        pl.col("numero_panneau").n_unique().alias("nb_listes_t1"),
        pl.col("bureau_de_vote").n_unique().alias("nb_bureaux"),
    )
    unites = comptes.filter(pl.col("nb_bureaux") >= pl.col("nb_listes_t1") + 2)[
        key
    ].to_list()

    abst1 = _abstention(resultats_t1, key).partition_by(key, as_dict=True)
    abst2 = _abstention(resultats_t2, key).partition_by(key, as_dict=True)
    t1_par_unite = resultats_t1.partition_by(key, as_dict=True)
    t2_par_unite = resultats_t2.partition_by(key, as_dict=True)

    lignes = []
    echecs = 0
    for unite in tqdm(unites):
        cle = (unite,)
        t1_values = _matrice(t1_par_unite[cle], abst1[cle])
        t2_values = _matrice(t2_par_unite[cle], abst2[cle])

        report = calculer_reports(t1_values, t2_values)
        if report is None:
            echecs += 1
            continue

        ligne: dict[str, object] = {
            key: unite,
            "coefficients": report.flatten("C"),
            "r_square": calculer_r_square(t1_values, t2_values, report),
        }
        if cv_splits:
            ligne["r_square_cv"] = calculer_r_square_validation(
                t1_values, t2_values, cv_splits
            )
        if bootstrap:
            resultat = bootstrap_reports(t1_values, t2_values, bootstrap)
            ligne["coefficients_std"] = (
                None if resultat is None else resultat[1].flatten("C")
            )
        lignes.append(ligne)

    pl.DataFrame(lignes).write_parquet(output)
    click.echo(
        f"{len(lignes)} unités traitées, {echecs} échecs sur {len(unites)} candidates."
    )


if __name__ == "__main__":
    main()
