# MMM Bayésien avec PyMC

Projet de Marketing Mix Modeling bayésien orienté production.

## En une phrase
Ce projet estime l'impact des investissements média sur les ventes avec un modèle bayésien PyMC, puis industrialise le flux complet (data pipeline, tuning, entraînement, métriques, artefacts, CI).

## Ce que le projet démontre
- Modélisation MMM bayésienne (incertitude, contributions, ROAS).
- Pipeline reproductible piloté par configuration YAML.
- Bonnes pratiques MLOps: tests, lint, CI, versionnement des artefacts.

## Stack
- Python, PyMC, ArviZ, NumPy, Pandas, scikit-learn
- Poetry, Pytest, Ruff, Black, GitHub Actions

## Exécution rapide
```bash
poetry install
poetry run mmm train --config configs/base.yaml
```

## Résultats produits
- Données préparées (`data/processed`)
- Inférence bayésienne (`artifacts/inference`)
- Métriques train/test (`artifacts/metrics`)
- Contributions, décomposition baseline/media, ROAS (`artifacts/*`)
- Figures de diagnostic (`artifacts/figures`)

## Détails techniques
Documentation technique complète: [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)
