# Identifying Plant Types using K-Means

**Unsupervised Machine Learning for Plant Classification**

Group plant images using **K-Means** on multiple feature representations, with full **experimental logging** for the CS 572 graduate final project.

## Project Overview

- Compare **raw pixels**, **handcrafted** (color histogram + HOG), and **CNN embeddings**
- Tune **k** with **silhouette**; report **ARI / NMI** vs taxonomy labels when available (diagnostic)
- **Reproducible runs:** `scripts/run_experiments.py` saves `metrics.json`, `metrics_table.csv`, figures, and `config_snapshot.json`

## Project Structure

```
plant_clustering/
├── README.md
├── LICENSE
├── CONTRIBUTION.md        # Contribution statement (paste into report)
├── CITATION.md            # Dataset & paper citations
├── requirements.txt
├── environment.yml
├── pytest.ini
├── config.py
├── data/
│   ├── raw/               # Local: trees/, bushes/, flowers/
│   └── inaturalist2021/   # Auto-downloaded iNaturalist 2021
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── clustering.py
│   ├── evaluation.py      # ARI, NMI, diagnostics, limitation notes
│   ├── experiment_runner.py
│   └── visualization.py
├── scripts/
│   ├── run_experiments.py    # Full experiment driver (use for report numbers)
│   └── generate_figures.py   # Regenerate figs from metrics.json
├── tests/                 # pytest
├── reports/
│   └── CS572_Grad_Final_Report.md   # Paper-style template (Grad)
├── results/               # Figures + experiment runs (see .gitignore)
└── notebooks/
    └── plant_clustering_pipeline.ipynb
```

## Setup

**pip:**
```bash
pip install -r requirements.txt
```

**conda:**
```bash
conda env create -f environment.yml
conda activate plant_clustering
```

PyTorch: [pytorch.org](https://pytorch.org) if needed.

## Data

### iNaturalist 2021 (default)

- `config.DATASET_SOURCE = "inaturalist2021"`
- Plant-only (**Plantae**) subset; first run **downloads** to `data/inaturalist2021/` (~10 GB for `train_mini`)
- Cap: `INATURALIST_MAX_IMAGES` (default 10k) for feasible runs

### Local folders

- `data/raw/trees/`, `bushes/`, `flowers/` — set `DATASET_SOURCE = "local"`

## Running experiments (recommended for the report)

From the repo root:

```bash
# Full pipeline: metrics + CSV + figures under results/experiments/run_<timestamp>/
python scripts/run_experiments.py

# Fast smoke test (fewer images, smaller k grid)
python scripts/run_experiments.py --quick

# Skip raw-no-PCA ablation
python scripts/run_experiments.py --no-ablations
```

**Regenerate figures** after editing plotting code:

```bash
python scripts/generate_figures.py results/experiments/run_YYYYMMDD_HHMMSS
```

## Notebook

```bash
jupyter notebook notebooks/plant_clustering_pipeline.ipynb
```

Use the notebook for exploration; **cite numbers from `scripts/run_experiments.py`** in the paper for consistency.

## Tests

```bash
python -m pytest tests/ -v
```

## Reproducibility (rubric)

| Item | Location |
|------|----------|
| Random seed | `RANDOM_SEED` in `config.py` |
| Hyperparameters | `config.py` (k-range, PCA, K-Means) |
| Run snapshot | `results/experiments/run_*/config_snapshot.json` |
| Tables / metrics | `metrics.json`, `metrics_table.csv` |
| Figure regeneration | `scripts/generate_figures.py` |

## Graduate deliverable (paper)

1. Fill **[reports/CS572_Grad_Final_Report.md](reports/CS572_Grad_Final_Report.md)** (export to PDF with Pandoc/Word/LaTeX).
2. Copy contribution bullets from **[CONTRIBUTION.md](CONTRIBUTION.md)**.
3. Add citations per **[CITATION.md](CITATION.md)**.

## Rubric coverage (CS 572 Grad)

| Criterion | What we implemented |
|-----------|----------------------|
| **Technical (10)** | Modular pipeline, K-Means + feature extraction, tests, experiment driver |
| **Experiments (5)** | Baselines (3 features), silhouette + ARI/NMI, PCA ablation, limitation notes |
| **Contribution (5)** | CONTRIBUTION.md + template Discussion sections |
| **Reproducibility (3)** | Config, seeds, scripts, saved metrics, figure regeneration |
| **Deliverable (5)** | Grad report template + figure workflow |
| **GitHub (2)** | README, requirements, environment.yml, LICENSE, structure |

## Repository

- **GitHub:** [https://github.com/D3VTHSTVR/plant_clustering](https://github.com/D3VTHSTVR/plant_clustering)
- **Branching:** [BRANCHING.md](BRANCHING.md)

## License

[MIT License](LICENSE). Dataset terms apply to iNaturalist data ([CITATION.md](CITATION.md)).
