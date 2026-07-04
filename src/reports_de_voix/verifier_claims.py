from __future__ import annotations

import numpy as np
import polars as pl

from reports_de_voix.donnees import PARQUET


def _cell(df: pl.DataFrame, circo: str, source: str, dest: str) -> float | None:
    r = df.filter(
        (pl.col("circonscription") == circo)
        & (pl.col("source") == source)
        & (pl.col("destination") == dest)
    )["report"]
    return float(r[0]) if len(r) else None


def _config_stats(df: pl.DataFrame, config: str, source: str, dest: str) -> dict:
    v = df.filter(
        (pl.col("configuration") == config)
        & (pl.col("source") == source)
        & (pl.col("destination") == dest)
    )["report"].to_numpy()
    if not v.size:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mediane": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "part_sup_70": float(np.mean(v > 0.70)),
    }


def main() -> None:
    df = pl.read_parquet(PARQUET)
    print("=" * 70)
    print("CLAIM 1 — Somme 80-02 : NFP→NFP ~100 %, Reconquête→RN ~75 %")
    print(f"  NFP→NFP        = {_cell(df, '80-02', 'NFP', 'NFP'):.0%}")
    print(f"  Reconquête→RN  = {_cell(df, '80-02', 'Reconquête', 'RN'):.0%}")

    r2 = (
        df.group_by("circonscription")
        .agg(pl.col("r2_echantillon").mean())["r2_echantillon"]
        .to_numpy()
    )
    print("=" * 70)
    print(f"CLAIM 2 — R² moyen en échantillon ≈ 0,98 : {np.nanmean(r2):.4f}")

    print("=" * 70)
    print("CLAIM 3 — Duels ENS-HOR/RN : NFP→ENS-HOR ~90 %, toujours > 70 %")
    s = _config_stats(df, "ENS-HOR / RN", "NFP", "ENS-HOR")
    print(
        f"  médiane={s['mediane']:.0%}  min={s['min']:.0%}  "
        f"part>70%={s['part_sup_70']:.0%}  (n={s['n']})"
    )

    print("=" * 70)
    print("CLAIM 4 — Mobilisation abstentionnistes : centre NFP plus haut")
    for dest in ["NFP", "ENS-HOR", "LR", "RN"]:
        v = df.filter(
            (pl.col("source") == "non exprimés (T1)") & (pl.col("destination") == dest)
        )["report"].to_numpy()
        if v.size:
            print(
                f"  non-exprimés(T1)→{dest:8s} médiane={np.median(v):.1%} (n={v.size})"
            )

    print("=" * 70)
    print("CLAIM 5 — circonscriptions avec R² (par colonne) < 90 %")
    bas = (
        df.select("circonscription", "destination", "r2_echantillon")
        .unique()
        .filter(pl.col("r2_echantillon") < 0.90)
        .sort("r2_echantillon")
    )
    print(f"  {bas.height} couples (circ., destination) sous 0,90")
    print(bas.head(20))


if __name__ == "__main__":
    main()
