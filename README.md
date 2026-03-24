# Identifying Plant Types using K-Means

**Unsupervised Machine Learning for Plant Classification**

Classify plant images (trees, bushes, flowers, etc.) using **K-Means clustering** on different feature representations.

## Project Overview

From the proposal:
- **Compare clustering quality** using: raw pixels, handcrafted features, pretrained CNN embeddings
- **Evaluate** with silhouette score and visual cluster analysis
- **Focus** on plant type separation (trees, bushes, flowers)

## Pipeline (4 Phases)

| Phase | Tasks |
|-------|-------|
| **1. Setup** | Dataset selection, preprocessing |
| **2. Feature engineering** | Raw pixels, handcrafted (color + HOG), CNN embeddings |
| **3. Clustering** | K-Means, tune k, silhouette scores |
| **4. Evaluation** | Visualize clusters, compare methods, report |

## Project Structure

```
plant_clustering/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml        # Optional conda env
├── config.py              # Paths, dataset source, k range, seeds
├── data/
│   ├── raw/               # Local images: trees/, bushes/, flowers/
│   └── inaturalist2021/   # iNaturalist 2021 (auto-downloaded)
├── src/
│   ├── data_loader.py     # Load image paths
│   ├── features.py        # Raw, handcrafted, CNN feature extraction
│   └── clustering.py      # K-Means, silhouette, PCA
├── results/               # Plots, cluster galleries
└── notebooks/
    └── plant_clustering_pipeline.ipynb
```

## Setup

**pip (recommended):**
```bash
pip install -r requirements.txt
```

**conda (optional):**
```bash
conda env create -f environment.yml
conda activate plant_clustering
```

PyTorch: if installs fail, follow [pytorch.org](https://pytorch.org) for your OS/CUDA.

## Data

### Option 1: iNaturalist 2021 (default)

The project uses **iNaturalist 2021** by default. Plant images (kingdom Plantae) are automatically downloaded via torchvision on first run.

- **config.py**: `DATASET_SOURCE = "inaturalist2021"`
- **Version**: `2021_train_mini` (~500K images; plants subset ~200K)
- **Subsample**: `INATURALIST_MAX_IMAGES = 10000` for faster runs
- **Download**: First run downloads to `data/inaturalist2021/` (~10GB for train_mini)

### Option 2: Local images

Place images in `data/raw/<class>/`:
- `data/raw/trees/`
- `data/raw/bushes/`
- `data/raw/flowers/`

Set `config.DATASET_SOURCE = "local"` in config.py.

## Quick Start

1. Install: `pip install -r requirements.txt`
2. Open `notebooks/plant_clustering_pipeline.ipynb`
3. Run all cells

With iNaturalist (default): first run downloads the dataset; subsequent runs use the cached data. The notebook extracts features, runs K-Means, computes silhouette scores, and visualizes cluster samples.

## Reproducibility

- **Random seed:** `RANDOM_SEED` in `config.py` (used for subsampling and K-Means).
- **Paths:** All paths are under `config.py` (`DATA_DIR`, `INATURALIST_DIR`, `RESULTS_DIR`).
- **Figures:** Run the notebook end-to-end; plots are saved under `results/` (e.g. `silhouette_vs_k.png`, `cluster_gallery_cnn.png`).

## Rubric alignment (course final project)

| Criterion | How this repo supports it |
|-----------|---------------------------|
| **Technical** | Modular `src/` (data, features, clustering); notebook orchestrates the pipeline. |
| **Experiments** | Compare raw vs handcrafted vs CNN features; silhouette vs *k*; cluster galleries—extend with ablations (e.g. *k*, `INATURALIST_MAX_IMAGES`, PCA on/off) in the report. |
| **Contribution** | State your novelty vs proposal: e.g. plant-only iNaturalist subset, three feature families, quantitative + visual evaluation. |
| **Reproducibility** | This README, `config.py`, seeds, `requirements.txt`, `environment.yml`. |
| **Deliverable** | Grad: paper-style write-up citing figures from `results/`; Undergrad: optional site/demo + short report. |
| **GitHub** | README, `requirements.txt`, `environment.yml`, `LICENSE`, `.gitignore`, clear layout. |

## Repository

- **GitHub:** [https://github.com/D3VTHSTVR/plant_clustering](https://github.com/D3VTHSTVR/plant_clustering)
- **Branching:** See [BRANCHING.md](BRANCHING.md) (`prod` → main, `dev` integration, `vdev` / `ldev` for individual work).

## License

This project is released under the [MIT License](LICENSE). Dataset terms (e.g. [iNaturalist](https://github.com/visipedia/inat_comp)) apply to downloaded data.
