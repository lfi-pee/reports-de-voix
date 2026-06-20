from __future__ import annotations

from pathlib import Path

import numpy as np

from generation.donnees import charger
from reports_de_voix import calculer_r_square, calculer_reports

DATA = Path("/home/veesion/hexagonal/data/02_clean/elections")
T1 = DATA / "2024-legislatives-1-bureau_de_vote.parquet"
T2 = DATA / "2024-legislatives-2-bureau_de_vote.parquet"
CORR = DATA / "2024-legislatives-correspondances-bureau_de_vote-circonscription.csv"


def main() -> None:
    circos = charger(T1, T2, CORR)
    print(f"circonscriptions chargées : {len(circos)}")

    r2_moyens: list[float] = []
    resolues = 0
    somme = None
    for c in circos:
        report = calculer_reports(c.t1, c.t2, poids=c.poids)
        if report is None:
            continue
        resolues += 1
        r2 = calculer_r_square(c.t1, c.t2, report)
        r2_moyens.append(float(np.mean(r2)))
        if c.code == "80-02":
            somme = (c, report, r2)

    print(f"résolues : {resolues}/{len(circos)}")
    print(f"R² moyen en échantillon : {np.mean(r2_moyens):.4f}")
    print(f"part R² (moyenne circo) < 0.90 : {np.mean(np.array(r2_moyens) < 0.90):.2%}")

    if somme is not None:
        c, report, r2 = somme
        print("\n=== 80-02 (2e circ. Somme) ===")
        print("blocs T1 :", c.blocs_t1, "+ non_exprimes")
        print("blocs T2 :", c.blocs_t2, "+ non_exprimes")
        print("bureaux :", c.t1.shape[0])
        lignes = [*c.blocs_t1, "non_exprimes"]
        cols = [*c.blocs_t2, "non_exprimes"]
        complement = 1 - report.sum(axis=1, keepdims=True)
        mat = np.hstack([report, complement])
        print("colonnes :", cols)
        for nom, row in zip(lignes, mat, strict=False):
            print(nom.ljust(20), " ".join(f"{v:5.0%}" for v in row))


if __name__ == "__main__":
    main()
