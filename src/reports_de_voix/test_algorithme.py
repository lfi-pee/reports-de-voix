from __future__ import annotations

import numpy as np

from reports_de_voix import calculer_r_square, calculer_reports


def test_recupere_matrice_report_connue():
    rng = np.random.default_rng(0)

    # 2 choix au t1, 2 choix au t2 ; report connu (lignes sommant à 1)
    report_vrai = np.array([[0.7, 0.3], [0.2, 0.8]])

    # 30 bureaux, effectifs t1 aléatoires, t2 = t1 @ report (conservation parfaite)
    t1_values = rng.integers(50, 500, size=(30, 2)).astype(float)
    t2_values = t1_values @ report_vrai

    report_estime = calculer_reports(t1_values, t2_values)

    assert np.allclose(report_estime, report_vrai, atol=1e-3)
    assert np.allclose(
        calculer_r_square(t1_values, t2_values, report_estime), 1.0, atol=1e-3
    )
