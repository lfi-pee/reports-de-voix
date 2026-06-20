from __future__ import annotations

import numpy as np

from reports_de_voix import (
    bootstrap_reports,
    calculer_r_square,
    calculer_r_square_validation,
    calculer_reports,
    conditionnement,
)

REPORT_VRAI = np.array([[0.7, 0.3], [0.2, 0.8]])


def _donnees_parfaites(n_bureaux: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    t1_values = rng.integers(50, 500, size=(n_bureaux, 2)).astype(float)
    return t1_values, t1_values @ REPORT_VRAI


def test_recupere_matrice_report_connue():
    t1_values, t2_values = _donnees_parfaites(30)

    report_estime = calculer_reports(t1_values, t2_values)

    assert np.allclose(report_estime, REPORT_VRAI, atol=1e-3)
    assert np.allclose(
        calculer_r_square(t1_values, t2_values, report_estime), 1.0, atol=1e-3
    )


def test_ponderation_ne_casse_pas_la_recuperation():
    t1_values, t2_values = _donnees_parfaites(30)
    poids = 1.0 / np.sqrt(t1_values.sum(axis=1))

    report_estime = calculer_reports(t1_values, t2_values, poids=poids)

    assert np.allclose(report_estime, REPORT_VRAI, atol=1e-3)


def test_reports_none_si_sources_colineaires():
    # deuxième colonne = 2 × première : rang 1, système non identifiable
    t1_values = np.array([[100.0, 200.0], [150.0, 300.0], [80.0, 160.0]])
    t2_values = np.array([[120.0, 180.0], [200.0, 250.0], [90.0, 150.0]])

    assert calculer_reports(t1_values, t2_values) is None


def test_conditionnement_plus_eleve_quand_sources_correlees():
    rng = np.random.default_rng(3)
    independantes = rng.integers(50, 500, size=(40, 2)).astype(float)
    base = rng.integers(50, 500, size=(40, 1)).astype(float)
    correlees = np.hstack([base, base + rng.normal(0, 1, size=(40, 1))])

    assert conditionnement(correlees) > conditionnement(independantes)


def test_r_square_validation_parfait_hors_echantillon():
    t1_values, t2_values = _donnees_parfaites(50)

    r_square_cv = calculer_r_square_validation(t1_values, t2_values, n_splits=5)

    assert np.allclose(r_square_cv, 1.0, atol=1e-3)


def test_validation_penalise_le_surapprentissage():
    # 14 bureaux pour ~6 coefficients par colonne, et aucun lien réel t1 → t2 :
    # le R² en échantillon est gonflé, le R² hors échantillon l'est nettement moins
    rng = np.random.default_rng(2)
    t1_values = rng.integers(50, 500, size=(14, 5)).astype(float)
    t2_values = rng.integers(50, 500, size=(14, 5)).astype(float)

    report = calculer_reports(t1_values, t2_values)
    r_square_in = calculer_r_square(t1_values, t2_values, report).mean()
    r_square_cv = calculer_r_square_validation(t1_values, t2_values, n_splits=5).mean()

    assert r_square_cv < r_square_in


def test_validation_none_si_trop_peu_de_bureaux():
    t1_values, t2_values = _donnees_parfaites(2)

    assert calculer_r_square_validation(t1_values, t2_values, n_splits=5) is None


def test_bootstrap_coefficients_stables_sur_donnees_parfaites():
    t1_values, t2_values = _donnees_parfaites(50)

    moyenne, ecart_type = bootstrap_reports(t1_values, t2_values, n_resamples=30)

    assert np.allclose(moyenne, REPORT_VRAI, atol=1e-2)
    assert np.all(ecart_type < 1e-2)
