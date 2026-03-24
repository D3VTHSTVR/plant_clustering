# Contribution statement (for your report)

**Project:** Unsupervised plant image grouping with K-Means on iNaturalist 2021 (Plantae).

**What is new (vs using only one feature type):**
- A **controlled comparison** of three feature families: raw pixels, handcrafted (color histogram + HOG), and **ImageNet-pretrained CNN embeddings**, with the same K-Means pipeline and evaluation.
- **Quantitative evaluation** using silhouette for internal cluster quality and, when labels are available, **ARI / NMI** against taxonomy-derived reference labels (species or folder class).
- **Sensitivity analysis** via a PCA-on/off ablation for raw pixels and scripted reproduction (`scripts/run_experiments.py`).

**Design choices (justify in paper):**
- K selected by **silhouette sweep** over k (not elbow on inertia alone) to match the course emphasis on unsupervised metrics.
- **Plant-only** subset for iNaturalist 2021 to align with the stated problem (plant types).
- **Subsample cap** (`INATURALIST_MAX_IMAGES`) for computational feasibility; report runtime and data scale.

**Team roles:** Fill in who did dataset, PCA, preprocessing, clustering experiments, visualization, and writing.
