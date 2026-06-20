from __future__ import annotations

import numpy as np

from reports_de_voix import (
    bootstrap_reports,
    calculer_r_square,
    calculer_r_square_validation,
    calculer_reports,
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


def test_r_square_validation_parfait_hors_echantillon():
    t1_values, t2_values = _donnees_parfaites(50)

    r_square_cv = calculer_r_square_validation(t1_values, t2_values, n_splits=5)

    assert np.allclose(r_square_cv, 1.0, atol=1e-3)


def test_r_square_validation_none_si_unite_trop_petite():
    t1_values, t2_values = _donnees_parfaites(3)

    assert calculer_r_square_validation(t1_values, t2_values, n_splits=5) is None


def test_bootstrap_coefficients_stables_sur_donnees_parfaites():
    t1_values, t2_values = _donnees_parfaites(50)

    moyenne, ecart_type = bootstrap_reports(t1_values, t2_values, n_resamples=30)

    assert np.allclose(moyenne, REPORT_VRAI, atol=1e-2)
    assert np.all(ecart_type < 1e-2)
