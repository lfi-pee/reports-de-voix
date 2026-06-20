from __future__ import annotations

import cvxpy as cp
import numpy as np


def calculer_reports(
    t1_values: np.ndarray,
    t2_values: np.ndarray,
    solver: str = "CLARABEL",
    poids: np.ndarray | None = None,
) -> np.ndarray | None:
    """Calcule la matrice de report.

    :param t1_values: la matrice des votes de premier tour
    :param t2_values: la matrice des votes de second tour
    :param poids: pondération par bureau (ex. 1/√inscrits) appliquée aux résidus, pour
        homogénéiser la variance entre bureaux de tailles différentes
    :return: la matrice de report, ou None si le système est non identifiable ou si le
        solveur n'atteint pas l'optimum

    t1_values et t2_values ont :
    - autant de lignes que de bureaux de vote dans la commune
    - autant de colonnes que de choix possibles au tour correspondant

    La dernière colonne correspond au nombre de personnes ne s'étant pas exprimées.

    La matrice de report a autant de lignes que de choix de premier tour et autant de
    colonnes que de choix de second tour.
    """

    # sources de premier tour colinéaires : les coefficients ne sont pas identifiables
    if np.linalg.matrix_rank(t1_values) < t1_values.shape[1]:
        return None

    # pour prendre en compte les situations de radiation / ajout d'électeurs
    excedents = t1_values.sum(axis=1) - t2_values.sum(axis=1)

    N1 = t1_values.shape[1]
    N2 = t2_values.shape[1]

    t1_values = np.append(t1_values, np.maximum(-excedents, 0)[:, np.newaxis], axis=1)
    t2_values = np.append(t2_values, np.maximum(excedents, 0)[:, np.newaxis], axis=1)

    # La N2+1ème colonne se déduit des autres : chaque ligne doit sommer à 1
    matrice_report = cp.Variable(shape=(N1 + 1, N2), nonneg=True)
    constraints = [matrice_report.sum(axis=1) <= np.ones(N1 + 1)]

    matrice_report_complete = cp.hstack(
        [
            matrice_report,
            1 - matrice_report.sum(axis=1).reshape([N1 + 1, 1], order="C"),
        ]
    )

    residus = t1_values @ matrice_report_complete - t2_values
    if poids is not None:
        residus = cp.multiply(np.tile(poids[:, np.newaxis], (1, N2 + 1)), residus)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(residus)), constraints)
    problem.solve(solver=solver)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    if matrice_report.value is None:
        return None
    return matrice_report.value[:-1, :]


def conditionnement(t1_values: np.ndarray) -> float:
    """Conditionnement de la matrice des votes de premier tour.

    Une valeur élevée signale des sources quasi colinéaires : les coefficients restent
    estimables mais sont numériquement instables (le filtre nb_bureaux ≥ nb_listes + 2
    est nécessaire mais pas suffisant pour garantir l'identifiabilité).
    """
    return float(np.linalg.cond(t1_values))


def calculer_r_square(
    t1_values: np.ndarray, t2_values: np.ndarray, report: np.ndarray
) -> np.ndarray:
    predicted_t2 = t1_values @ report
    var_totale = t2_values.var(axis=0)
    var_residuels = (t2_values - predicted_t2).var(axis=0)
    var_totale = np.maximum(var_totale, 1e-10)
    return 1 - var_residuels / var_totale


def _ajuster_splits(n_bureaux: int, n_regresseurs: int, n_splits: int) -> int | None:
    """Plus grand nombre de plis (≤ n_splits) laissant chaque pli d'entraînement
    identifiable, ou None si même 2 plis sont insuffisants."""
    for k in range(min(n_splits, n_bureaux), 1, -1):
        taille_train = n_bureaux - -(-n_bureaux // k)  # n − ceil(n/k)
        if taille_train >= n_regresseurs:
            return k
    return None


def calculer_r_square_validation(
    t1_values: np.ndarray,
    t2_values: np.ndarray,
    n_splits: int = 5,
    solver: str = "CLARABEL",
    seed: int = 0,
    poids: np.ndarray | None = None,
) -> np.ndarray | None:
    """R² hors échantillon, estimé par validation croisée sur les bureaux de vote.

    Contrairement au R² en échantillon, cet indicateur sanctionne le surapprentissage :
    la matrice est ajustée sur une partie des bureaux puis évaluée sur les bureaux
    laissés de côté. Le nombre de plis est réduit automatiquement pour les petites
    unités ; None est renvoyé si même deux plis sont sous-déterminés.
    """

    n_bureaux = t1_values.shape[0]
    k = _ajuster_splits(n_bureaux, t1_values.shape[1], n_splits)
    if k is None:
        return None

    rng = np.random.default_rng(seed)
    ordre = rng.permutation(n_bureaux)
    predictions = np.full_like(t2_values, np.nan, dtype=float)

    for pli in np.array_split(ordre, k):
        train = np.setdiff1d(ordre, pli)
        poids_train = None if poids is None else poids[train]
        report = calculer_reports(
            t1_values[train], t2_values[train], solver=solver, poids=poids_train
        )
        if report is None:
            return None
        predictions[pli] = t1_values[pli] @ report

    var_totale = np.maximum(t2_values.var(axis=0), 1e-10)
    var_residuels = (t2_values - predictions).var(axis=0)
    return 1 - var_residuels / var_totale


def bootstrap_reports(
    t1_values: np.ndarray,
    t2_values: np.ndarray,
    n_resamples: int = 200,
    solver: str = "CLARABEL",
    seed: int = 0,
    poids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estime l'incertitude par cellule de la matrice de report.

    Rééchantillonne les bureaux de vote avec remise et renvoie la moyenne et l'écart
    type des matrices obtenues. Un écart type élevé signale un coefficient instable,
    même lorsque le R² global est proche de 1 (problème d'identifiabilité). Renvoie
    None si aucun rééchantillon ne donne de solution.
    """

    n_bureaux = t1_values.shape[0]
    rng = np.random.default_rng(seed)
    echantillons = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n_bureaux, size=n_bureaux)
        poids_idx = None if poids is None else poids[idx]
        report = calculer_reports(
            t1_values[idx], t2_values[idx], solver=solver, poids=poids_idx
        )
        if report is not None:
            echantillons.append(report)

    if not echantillons:
        return None

    pile = np.stack(echantillons)
    return pile.mean(axis=0), pile.std(axis=0)
