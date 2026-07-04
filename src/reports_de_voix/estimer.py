from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import polars as pl
from tqdm import tqdm

from reports_de_voix import (
    calculer_r_square,
    calculer_r_square_validation,
    calculer_reports,
    conditionnement,
)
from reports_de_voix.algorithme import echantillons_bootstrap
from reports_de_voix.donnees import DATA, PARQUET, Circonscription, charger

# Blocs retenus pour étiqueter une configuration de second tour (qualifiés au sens
# courant : NFP / centre-droit présidentiel / droite / extrême droite).
BLOCS_CONFIG = ["NFP", "ENS-HOR", "LR", "RN", "Union ext. droite"]


@dataclass(frozen=True)
class Reglages:
    n_bootstrap: int = 300
    niveau: float = 0.95
    seed: int = 0


def _configuration(blocs_t2: list[str]) -> str:
    presents = [b for b in BLOCS_CONFIG if b in blocs_t2]
    return " / ".join(presents) if presents else "autre"


def _bootstrap_percentiles(
    c: Circonscription, reglages: Reglages
) -> tuple[np.ndarray, np.ndarray] | None:
    # Percentiles bootstrap sur le rééchantillonnage des bureaux : ils mesurent la
    # stabilité de l'estimation, non une couverture à 95 %. L'estimateur est contraint
    # à [0, 1] et le percentile bootstrap n'est pas valide au bord (Andrews, 2000) ;
    # d'où « plage de stabilité » plutôt qu'« intervalle de confiance ».
    pile = echantillons_bootstrap(
        c.t1, c.t2, reglages.n_bootstrap, seed=reglages.seed, poids=c.poids
    )
    if pile is None or len(pile) < reglages.n_bootstrap // 2:
        return None
    alpha = (1 - reglages.niveau) / 2
    bas = np.quantile(pile, alpha, axis=0)
    haut = np.quantile(pile, 1 - alpha, axis=0)
    return bas, haut


def estimer_une(c: Circonscription, reglages: Reglages) -> list[dict[str, object]]:
    report = calculer_reports(c.t1, c.t2, poids=c.poids)
    if report is None:
        return []
    r2 = calculer_r_square(c.t1, c.t2, report)
    r2_cv = calculer_r_square_validation(c.t1, c.t2, poids=c.poids)
    cond = conditionnement(c.t1)
    config = _configuration(c.blocs_t2)
    bornes = _bootstrap_percentiles(c, reglages)
    sources = [*c.blocs_t1, "non exprimés (T1)"]
    dests = [*c.blocs_t2, "non exprimés (T2)"]
    lignes: list[dict[str, object]] = []
    for i, source in enumerate(sources):
        for j, dest in enumerate(dests):
            lignes.append(
                {
                    "circonscription": c.code,
                    "configuration": config,
                    "n_bureaux": c.t1.shape[0],
                    "source": source,
                    "destination": dest,
                    "report": float(report[i, j]),
                    "stabilite_bas": None if bornes is None else float(bornes[0][i, j]),
                    "stabilite_haut": None
                    if bornes is None
                    else float(bornes[1][i, j]),
                    "r2_echantillon": float(r2[j]) if j < len(r2) else None,
                    "r2_validation": None
                    if r2_cv is None
                    else (float(r2_cv[j]) if j < len(r2_cv) else None),
                    "conditionnement": cond,
                }
            )
    return lignes


def _init_worker() -> None:
    # un solveur convexe par circonscription : on évite la sur-souscription des
    # threads BLAS quand plusieurs processus tournent en parallèle.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def estimer(
    circos: list[Circonscription], reglages: Reglages, n_jobs: int | None = None
) -> pl.DataFrame:
    n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
    lignes: list[dict[str, object]] = []
    if n_jobs == 1:
        for c in tqdm(circos):
            lignes.extend(estimer_une(c, reglages))
    else:
        _init_worker()
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            mp_context=mp.get_context("fork"),
        ) as pool:
            futures = [pool.submit(estimer_une, c, reglages) for c in circos]
            for f in tqdm(as_completed(futures), total=len(futures)):
                lignes.extend(f.result())
    return pl.DataFrame(lignes)


def main() -> None:
    circos = charger(
        DATA / "2024-legislatives-1-bureau_de_vote.parquet",
        DATA / "2024-legislatives-2-bureau_de_vote.parquet",
        DATA / "2024-legislatives-correspondances-bureau_de_vote-circonscription.csv",
    )
    print(f"circonscriptions : {len(circos)} ; processus : {os.cpu_count()}")
    df = estimer(circos, Reglages())
    df.write_parquet(PARQUET)
    print(f"écrit : {PARQUET}  ({df.height} cellules)")


if __name__ == "__main__":
    main()
