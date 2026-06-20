from __future__ import annotations

import cvxpy as cp
import numpy as np


def calculer_reports(
    t1_values: np.ndarray, t2_values: np.ndarray, solver: str = "CLARABEL"
) -> np.ndarray | None:
    """Calcule la matrice de report.

    :param t1_values: la matrice des votes de premier tour
    :param t2_values: la matrice des votes de second tour
    :return: la matrice de report

    t1_values et t2_values ont :
    - autant de lignes que de bureaux de vote dans la commune
    - autant de colonnes que de choix possibles au tour correspondant

    La dernière colonne correspond au nombre de personnes s'étant abstenu.

    La matrice de report a autant de lignes que de choix de premier tour et autant de
    colonnes que de choix de second tour.
    """

    # pour prendre en compte les situations de radiation / ajout d'électeurs
    excedents = t1_values.sum(axis=1) - t2_values.sum(axis=1)

    n1 = t1_values.shape[1]
    n2 = t2_values.shape[1]

    t1_values = np.append(t1_values, np.maximum(-excedents, 0)[:, np.newaxis], axis=1)
    t2_values = np.append(t2_values, np.maximum(excedents, 0)[:, np.newaxis], axis=1)

    # La (n2+1)ème colonne se déduit des autres : chaque ligne doit sommer à 1
    matrice_report = cp.Variable(shape=(n1 + 1, n2), nonneg=True)
    constraints = [matrice_report.sum(axis=1) <= np.ones(n1 + 1)]

    matrice_report_complete = cp.hstack(
        [
            matrice_report,
            1 - matrice_report.sum(axis=1).reshape([n1 + 1, 1], order="C"),
        ]
    )

    prediction = t1_values @ matrice_report_complete

    objective = cp.Minimize(cp.sum_squares(prediction - t2_values))

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver)

    if matrice_report.value is not None:
        return matrice_report.value[:-1, :]


def calculer_r_square(
    t1_values: np.ndarray, t2_values: np.ndarray, report: np.ndarray
) -> np.ndarray:
    predicted_t2 = t1_values @ report
    var_totale = t2_values.var(axis=0)
    var_residuels = (t2_values - predicted_t2).var(axis=0)
    var_totale = np.maximum(var_totale, 1e-10)
    return 1 - var_residuels / var_totale
