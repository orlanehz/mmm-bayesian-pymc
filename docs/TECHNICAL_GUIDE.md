# Documentation Technique - MMM Bayésien (PyMC)

## 1. Objectif
Construire un pipeline MMM bayésien de bout en bout:
- ingestion et préparation des données,
- feature engineering (adstock/saturation/saisonnalité),
- tuning rapide des hyperparamètres média,
- entraînement PyMC final,
- évaluation et production d'artefacts business.

## 2. Architecture du projet
```text
src/mmm/
  cli.py              # point d'entrée CLI
  train.py            # orchestration E2E
  config.py           # chargement config YAML
  data_io.py          # ingestion, dérivations, validation
  validation.py       # contrôles qualité des datasets
  transforms.py       # adstock + saturation
  features.py         # matrice X/y pour le modèle
  split.py            # split temporel
  tuning.py           # recherche des decay via Ridge
  model.py            # modèle bayésien PyMC
  predict.py          # posterior predictive sur nouvelles X
  evaluation.py       # métriques et transformations
  contributions.py    # contributions média
  decomposition.py    # baseline vs media
  roas.py             # calcul de ROAS
  plotting.py         # graphiques
```

## 3. Pipeline d'entraînement (`mmm train`)
Commande:
```bash
poetry run mmm train --config configs/base.yaml
```

Étapes exécutées:
1. `run_data_pipeline`:
- lit la source weekly,
- convertit la date,
- dérive un daily dataset,
- ré-agrège en weekly modélisation,
- valide le schéma final.

2. `tune_adstock_decays`:
- grid search sur les `decay_candidates`,
- modèle Ridge rapide,
- sélection du meilleur set de décays selon `rmse` ou `mape`.

3. Fit final bayésien (`fit_mmm`):
- split temporel train/test,
- construction design matrix,
- inférence MCMC PyMC,
- posterior predictive.

4. Évaluation:
- métriques train/test,
- contributions média,
- décomposition baseline/media,
- ROAS.

5. Sauvegarde des artefacts (NetCDF, parquet, JSON, figures).

## 4. Modèle PyMC (version actuelle)
Forme:
- `y ~ Normal(mu, sigma)`
- `mu = intercept + X @ beta`

Priors:
- `intercept ~ Normal(0, 2)`
- `beta ~ Normal(0, 1)`
- `sigma ~ HalfNormal(1)`

Notes:
- `X` intègre médias transformés, contrôles, saisonnalité Fourier.
- target transform configurable (`log1p`/`none`).

### 4.1 Variables de contrôle (exemples)
Les variables de contrôle sont des facteurs non-média qui influencent `y`.
Elles sont incluses dans `X` au même titre que les autres features, pour éviter
d'attribuer à tort leur effet aux canaux média.

Exemples fréquents:
- `covid_lockdown` (0/1 pendant les périodes de confinement),
- `boycott_event` (0/1 pendant un boycott),
- `stockout_rate` (taux de rupture),
- `promo_intensity` / `avg_price` (pression promo et prix),
- météo extrême, jours fériés, vacances scolaires.

Point important:
- on ne retire pas ces variables du modèle;
- elles servent à isoler l'effet média de l'effet contexte;
- le ROAS est ensuite calculé sur la composante média.

### 4.2 Saisonnalité: comment elle intervient dans le calcul
La saisonnalité n'est pas un scalaire fixe (ni une simple pénalité négative).
Elle est modélisée par des colonnes supplémentaires dans `X` (termes de Fourier,
ex: `sin`/`cos`), avec leurs propres coefficients `beta`.

Dans `mu = intercept + X @ beta`, une partie de `X @ beta` correspond donc
à la contribution saisonnière.

Conséquences:
- selon la semaine, la contribution saisonnière peut être positive, négative ou nulle;
- elle varie dans le temps de façon cyclique;
- elle évite d'attribuer aux médias des variations qui viennent du calendrier.

Exemples:
- semaine de Noël: contribution saisonnière souvent positive (hausse structurelle),
- semaine creuse hors saison: contribution saisonnière souvent négative,
- semaine \"normale\": contribution proche de zéro.

En pratique, l'effet total prédit est une somme:
- niveau de base (`intercept`),
- effet médias transformés,
- effet contrôles,
- effet saisonnalité,
- bruit résiduel (`sigma`).

### 4.3 Scénarios, courbe de réponse et uplift (exemple concret)
Idée:
- la courbe de réponse convertit un niveau de spend en contribution attendue,
- un scénario modifie le spend (ex: `+20%` sur un canal),
- l'uplift mesure le gain incrémental vs baseline.

Formule:
- `uplift = y_pred(scenario) - y_pred(baseline)`

Exemple simple (1 semaine):
- baseline: spend TV = `100`, contribution média prédite = `30`, ventes prédites = `130`
- scénario S1: spend TV = `120` (`+20%`)
- via la courbe de réponse (avec saturation), contribution TV passe de `30` à `34` (pas `36`, car rendements décroissants)
- ventes prédites scénario = `134`
- uplift = `134 - 130 = +4`

Interprétation:
- `+20%` de budget TV apporte `+4` ventes incrémentales dans ce contexte,
- le gain marginal est plus faible à haut niveau de spend (effet saturation),
- on compare plusieurs scénarios pour arbitrer l'allocation budgétaire.

## 5. Configuration (`configs/base.yaml`)
Sections principales:
- `paths`: emplacements data/artifacts
- `data`: mapping colonnes source/processed
- `features`: saisonnalité, transformations
- `split`: taille du test set
- `training`: paramètres MCMC (`draws`, `tune`, `chains`)
- `tuning`: espace de recherche des décays

Contrôles auto-détectés:
- si `source_control_cols: []`, détection des colonnes contenant `hldy_`
- si `processed_control_cols: []`, reprise de la même liste

## 6. Validation et garde-fous
`validation.py` vérifie:
- présence des colonnes attendues,
- date triée/unique/non nulle,
- non-négativité target/channels,
- type numérique des contrôles.

## 7. Tests
Commandes:
```bash
poetry run pytest -q
poetry run ruff check src tests
```

Couverture (principale):
- transformations (`test_transforms.py`)
- validation data (`test_validation.py`)
- features et split (`test_features.py`, `test_split.py`)
- pipeline data invariants (`test_data_pipeline.py`)
- smoke test E2E (`test_pipeline_smoke.py`)
- tuning (`test_tuning.py`)

## 8. Artefacts produits
Sous `artifacts/`:
- `inference/`: `idata.nc`, metadata
- `metrics/`: métriques JSON
- `tuning/`: résultats tuning
- `contributions/`: contributions média
- `decomposition/`: baseline vs media
- `roas/`: synthèse ROAS
- `figures/`: PNG de visualisation

## 9. Industrialisation
- CLI unique (`mmm train`) pour exécution reproductible
- dépendances verrouillées avec Poetry (`poetry.lock`)
- qualité code: Ruff + Black + Pytest
- CI GitHub Actions (`.github/workflows/ci.yml`)
- séparation claire `src/`, `configs/`, `data/`, `artifacts/`, `tests/`

## 10. Limitations connues et pistes d'amélioration
- modèle linéaire bayésien simple (pas encore hiérarchique)
- prior engineering à enrichir (par canal)
- calibration business avancée ROAS/contributions à renforcer
- monitoring de drift et retraining non automatisés (prochaine étape)
