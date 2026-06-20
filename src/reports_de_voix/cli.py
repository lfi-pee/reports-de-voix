from __future__ import annotations

import click
import polars as pl
from tqdm import tqdm

from reports_de_voix.algorithme import calculer_r_square, calculer_reports


@click.command()
@click.option("--t1", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--t2", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--key", required=True)
@click.option(
    "-o", "--output", type=click.Path(writable=True, dir_okay=False), required=True
)
def main(t1: str, t2: str, key: str, output: str) -> None:
    resultats_t1 = pl.read_parquet(t1)
    resultats_t2 = pl.read_parquet(t2)

    resultats_t1 = resultats_t1.with_columns(
        bureau_de_vote=pl.format("{code_commune}-{bureau_de_vote}")
    )
    resultats_t2 = resultats_t2.with_columns(
        bureau_de_vote=pl.format("{code_commune}-{bureau_de_vote}")
    )

    unites = (
        resultats_t2.group_by(key)
        .agg(
            pl.col("numero_panneau").n_unique().alias("nb_listes_t2"),
            pl.col("bureau_de_vote").n_unique().alias("nb_bureaux"),
        )
        .join(
            resultats_t1.group_by(key).agg(
                pl.col("numero_panneau").n_unique().alias("nb_listes_t1"),
            ),
            on=key,
            validate="1:1",
        )
    )

    abstention_t1 = (
        resultats_t1.group_by("bureau_de_vote")
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

    abstention_t2 = (
        resultats_t2.group_by("bureau_de_vote")
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

    # on veut au moins autant de bureaux que de variables
    unites = unites.filter(
        pl.col("nb_listes_t1") * pl.col("nb_listes_t1") <= pl.col("nb_bureaux")
    )[key]

    report_par_unite = {}

    for unite in tqdm(unites):
        t2_unite = (
            resultats_t2.sort(["bureau_de_vote", "numero_panneau"])
            .filter(pl.col(key) == unite)
            .pivot(
                on=["numero_panneau"],
                index=["bureau_de_vote"],
                values=["voix"],
                maintain_order=True,
            )
            .join(
                abstention_t2.filter(pl.col(key) == unite).select(
                    "bureau_de_vote", "abstention"
                ),
                on=["bureau_de_vote"],
                validate="1:1",
                maintain_order="left",
            )
        )

        t1_unite = (
            resultats_t1.sort(["bureau_de_vote", "numero_panneau"])
            .filter(pl.col(key) == unite)
            .pivot(
                on=["numero_panneau"],
                index=["bureau_de_vote"],
                values=["voix"],
                maintain_order=True,
            )
            .join(
                abstention_t1.filter(pl.col(key) == unite).select(
                    "bureau_de_vote", "abstention"
                ),
                on=["bureau_de_vote"],
                validate="1:1",
                maintain_order="left",
            )
        )

        t2_values = t2_unite.select(pl.all().exclude(["bureau_de_vote"])).to_numpy()
        t1_values = t1_unite.select(pl.all().exclude(["bureau_de_vote"])).to_numpy()

        matrice_report = calculer_reports(t1_values, t2_values)

        if matrice_report is not None:
            r_square = calculer_r_square(t1_values, t2_values, matrice_report)

            report_par_unite[unite] = (matrice_report.flatten("C"), r_square)

    df_reports = pl.DataFrame(
        {
            key: report_par_unite.keys(),
            "coefficients": [c for c, _ in report_par_unite.values()],
            "r_square": [r for _, r in report_par_unite.values()],
        }
    )

    df_reports.write_parquet(output)


if __name__ == "__main__":
    main()
