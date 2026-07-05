# Reports de voix

Ce dépôt produit le rapport « Analyse des reports entre premier et deuxième tours
des législatives 2024 » ([`rapport.ipynb`](rapport.ipynb)), et fournit la
bibliothèque et la ligne de commande d'estimation sous-jacentes.

L'estimation des **reports de voix** entre deux tours se fait par optimisation
convexe sur les résultats au niveau des bureaux de vote. Pour chaque unité
géographique (commune, circonscription…), on cherche la matrice de report `R` qui
minimise l'écart entre les voix observées au second tour et leur prédiction à partir
du premier tour :

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

## Le rapport

Le notebook [`rapport.ipynb`](rapport.ipynb) est le rapport : la prose y est
entrecoupée de cellules qui recalculent chaque tableau et chaque graphe depuis
`data/reports_legislatives_2024.parquet`. Le ré-exécuter suffit à tout rafraîchir
(le groupe de dépendances `dev` fournit le noyau Jupyter).

- `uv run python -m reports_de_voix.estimer` régénère le parquet depuis les
  résultats par bureau de vote du dépôt `hexagonal` : matrices de report par
  circonscription, R² en échantillon et de validation croisée, plages de stabilité
  bootstrap ;
- `uv run python -m reports_de_voix.verifier_claims` confronte les chiffres cités
  dans la prose au contenu du parquet.

Les modules du rapport sont `donnees` (chargement et nuançage en blocs), `estimer`
(pipeline d'estimation) et `graphiques` (rendu Altair des panneaux de reports) ; les
tableaux et graphes eux-mêmes sont calculés directement dans les cellules du notebook.

## En ligne de commande

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
matrice de report aplatie (ordre ligne par ligne), le `r_square` (en échantillon) par
colonne de second tour, et le `conditionnement` de la matrice des votes de premier tour
(une valeur élevée signale des sources quasi colinéaires, donc des coefficients
numériquement instables).

L'ajustement est pondéré par bureau (`1/√inscrits`) pour homogénéiser la variance : sans
pondération, les gros bureaux domineraient mécaniquement l'estimation. La dernière
colonne implicite capte les électeurs **non exprimés** (abstention, mais aussi votes
blancs et nuls).

Deux options ajoutent des indicateurs de fiabilité :

- `--cv-splits N` (défaut `5`, `0` pour désactiver) calcule un `r_square_cv` par
  validation croisée sur les bureaux. C'est le R² **hors échantillon** : il teste
  réellement l'hypothèse d'homogénéité, contrairement au `r_square` en échantillon qui
  est largement mécanique sur un modèle aussi paramétré.
- `--bootstrap N` (défaut `0`) ajoute `coefficients_std`, l'écart type de chaque cellule
  obtenu en rééchantillonnant les bureaux. Un écart type élevé signale un coefficient
  instable (non identifiable) même quand le R² est proche de 1.

## En bibliothèque

```python
from reports_de_voix import calculer_reports, calculer_r_square
```

## Notes

Une unité n'est traitée que si elle compte au moins autant de bureaux de vote que de
coefficients à estimer par colonne de second tour, soit
`nb_bureaux ≥ nb_listes_t1 + 2` (les listes de premier tour, plus les non-exprimés et
les excédents/déficits d'inscrits), faute de quoi le système est sous-déterminé. Ce
filtre est nécessaire mais pas suffisant : des sources quasi colinéaires restent
instables, ce que révèlent le `conditionnement` et l'écart type bootstrap. Les sources
exactement colinéaires (rang déficient) sont écartées.
