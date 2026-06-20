# Reports de voix

Estimation des **reports de voix** entre deux tours d'une élection, par optimisation
convexe sur les résultats au niveau des bureaux de vote.

Pour chaque unité géographique (commune, circonscription…), on cherche la matrice de
report `R` qui minimise l'écart entre les voix observées au second tour et leur
prédiction à partir du premier tour :

```
minimiser  || T1 · R − T2 ||²
sous        R ≥ 0  et  somme de chaque ligne ≤ 1
```

Chaque ligne de `R` correspond à un choix de premier tour (liste/candidat, plus
l'abstention), chaque colonne à un choix de second tour. La colonne implicite
restante capte le report vers l'abstention. Les excédents/déficits d'inscrits entre
les deux tours (radiations, ajouts) sont absorbés par une colonne dédiée.

Extrait du dépôt [`hexagonal`](https://github.com/lfi-pee/hexagonal).

## Installation

L'installation se fait avec le gestionnaire de paquets
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

## Utilisation

Les entrées sont deux fichiers Parquet de résultats par bureau de vote (un par tour),
contenant au minimum les colonnes `code_commune`, `bureau_de_vote`, `numero_panneau`,
`voix`, `inscrits`, ainsi que la colonne clé d'agrégation (`--key`).

```bash
uv run reports-de-voix \
  --key code_commune \
  --t1 2026-municipales-1-bureau_de_vote.parquet \
  --t2 2026-municipales-2-bureau_de_vote.parquet \
  --output 2026-municipales-reports.parquet
```

La sortie est un Parquet avec une ligne par unité : la clé, les `coefficients` de la
matrice de report aplatie (ordre ligne par ligne), et le `r_square` (en échantillon)
par colonne de second tour.

Deux options ajoutent des indicateurs de fiabilité :

- `--cv-splits N` (défaut `5`, `0` pour désactiver) calcule un `r_square_cv` par
  validation croisée sur les bureaux. C'est le R² **hors échantillon** : il teste
  réellement l'hypothèse d'homogénéité, contrairement au `r_square` en échantillon qui
  est largement mécanique sur un modèle aussi paramétré.
- `--bootstrap N` (défaut `0`) ajoute `coefficients_std`, l'écart type de chaque cellule
  obtenu en rééchantillonnant les bureaux. Un écart type élevé signale un coefficient
  instable (non identifiable) même quand le R² est proche de 1.

### En bibliothèque

```python
from reports_de_voix import calculer_reports, calculer_r_square
```

## Notes

Une unité n'est traitée que si elle compte au moins autant de bureaux de vote que de
coefficients à estimer par colonne de second tour, soit
`nb_bureaux ≥ nb_listes_t1 + 2` (les listes de premier tour, plus l'abstention et les
excédents/déficits d'inscrits), faute de quoi le système est sous-déterminé.
