# Identifying Plant Types with K-Means on iNaturalist 2021

**Course:** CS 572 — Unsupervised Machine Learning (Graduate)  
**Authors:** [Your Name], [Colleague Name]  
**Date:** Spring 2026

---

## Abstract

*(150–250 words.) Summarize: problem (unsupervised grouping of plant images), data (iNaturalist 2021, Plantae), method (K-Means on three feature types), metrics (silhouette; ARI/NMI vs taxonomy), key finding (which feature works best and why), and one limitation.*

**Keywords:** unsupervised learning, K-Means, clustering, iNaturalist, deep features, silhouette

---

## 1. Introduction

1.1 **Motivation** — Manual plant identification is time-consuming; large ecological image collections need structure without exhaustive labels.

1.2 **Problem** — Given only images, discover **natural groupings** that reflect plant appearance (trees vs herbs vs flowers, etc.) without training a supervised classifier.

1.3 **Contributions** — (See [CONTRIBUTION.md](../CONTRIBUTION.md) in repo; paste a shortened version here.)

---

## 2. Related Work

- Briefly cite **iNaturalist benchmarks** and **FGVC / iNat Challenge** literature.
- Cite **K-Means** and **clustering evaluation** (silhouette, ARI/NMI when reference labels exist).
- Cite **transfer learning** for CNN embeddings used as features.
- References: use [CITATION.md](../CITATION.md) and add your own BibTeX entries.

---

## 3. Method

### 3.1 Data

- **Dataset:** iNaturalist 2021 (`train_mini` or full train), **kingdom Plantae** only.
- **Preprocessing:** Resize for raw/handcrafted (see `config.IMG_SIZE`); 224×224 for CNN embeddings.
- **Subsample:** State `INATURALIST_MAX_IMAGES` and random seed for reproducibility.

### 3.2 Features

| Feature | Description |
|---------|-------------|
| Raw pixels | Vectorized RGB after resize |
| Handcrafted | RGB histograms + HOG (texture/shape cues) |
| CNN embeddings | Final pooling layer of pretrained ResNet (see `CNN_BACKBONE` in config) |

### 3.3 Clustering

- **Algorithm:** K-Means (`k-means++` init).
- **Number of clusters *k*:** Grid search over `K_RANGE`; select **best k by mean silhouette** on the feature space used for clustering.
- **PCA:** Applied to raw-pixel features before K-Means when enabled (variance retained in config); report as ablation.

### 3.4 Evaluation

- **Unsupervised:** Silhouette (global), silhouette per cluster (for failure analysis).
- **Supervised (diagnostic):** ARI and NMI between cluster IDs and **species-level** folder IDs (iNaturalist) or folder labels (local data). *Note:* high species cardinality makes perfect alignment unlikely; interpret as **partial alignment**, not accuracy.

---

## 4. Experiments

### 4.1 Protocol

- List **commands** from README: `python scripts/run_experiments.py` (and `--quick` for development).
- State **hardware** (CPU/GPU, RAM) and **approximate runtime**.

### 4.2 Main Results

**Table 1:** Copy from `results/experiments/run_*/metrics_table.csv` or paste here.

| Feature | Best k | Silhouette | ARI | NMI |
|---------|--------|------------|-----|-----|
| raw_pixels | | | | |
| handcrafted | | | | |
| cnn_embeddings | | | | |

**Figures:** Insert from `results/experiments/run_*/`:
- `fig_silhouette_vs_k.png`
- `fig_feature_silhouette_bar.png`
- Cluster galleries from notebook (`cluster_gallery_cnn.png`).

### 4.3 Ablations / Sensitivity

- **Raw + PCA vs raw without PCA** — compare silhouette and stability (from `ablations` in `metrics.json`).
- **Optional:** Vary `INATURALIST_MAX_IMAGES` (e.g., 2k vs 10k) and report how metrics change.

### 4.4 Limitations & Error Analysis

Use auto-generated `limitation_notes` in `metrics.json` plus qualitative discussion:
- K-Means assumes spherical clusters; species may not match visual clusters.
- **Species vs cluster:** Many species per visual “type” may lower ARI/NMI even when clusters look visually coherent.
- **Sampling:** Subsample may under-represent rare species.

---

## 5. Discussion

- Which feature representation best matches your **visual** inspection of clusters?
- Trade-off: **interpretability** vs **metric** (silhouette vs taxonomy alignment).
- When would you recommend handcrafted vs CNN features for field deployment?

---

## 6. Conclusion

- Restate findings in one paragraph.
- Future work: hierarchical clustering, constrained K-Means, self-supervised embeddings.

---

## References

1. Van Horn et al., iNaturalist competition / dataset overview.  
2. MacQueen, K-Means.  
3. Rousseeuw, Silhouette.  
4. He et al., ResNet (if using ResNet embeddings).  
5. iNaturalist terms of use / dataset citation.

*(Expand with BibTeX in your final PDF.)*

---

## Appendix A. Reproducibility

- **Seed:** `RANDOM_SEED` in `config.py`
- **Config snapshot:** `config_snapshot.json` in each run directory
- **Regenerate figures:** `python scripts/generate_figures.py results/experiments/run_<id>`

---

## Appendix B. Division of Labor

| Task | Person |
|------|--------|
| Dataset / preprocessing | |
| PCA / features | |
| Clustering experiments | |
| Figures / report | |
