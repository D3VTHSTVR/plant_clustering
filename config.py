"""
Configuration for Identifying Plant Types using K-Means
Unsupervised Machine Learning for Plant Classification
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
INATURALIST_DIR = DATA_DIR / "inaturalist2021"  # Downloaded iNaturalist 2021

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, INATURALIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset source
# ---------------------------------------------------------------------------
# "local" = use images in data/raw/ (trees/, bushes/, flowers/)
# "inaturalist2021" = use iNaturalist 2021 plant images (downloads via torchvision)
DATASET_SOURCE = "inaturalist2021"

# iNaturalist 2021 settings (when DATASET_SOURCE == "inaturalist2021")
INATURALIST_VERSION = "2021_train_mini"  # 500K images; use "2021_train" for full 2.7M
INATURALIST_PLANTS_ONLY = True           # Filter for kingdom Plantae
INATURALIST_MAX_IMAGES = 10000           # Subsample for manageable runs (None = use all plants)

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMG_SIZE = (64, 64)       # Smaller for raw pixels (64*64*3 = 12,288); increase for CNN
IMG_SIZE_CNN = (224, 224) # Standard size for pretrained CNN embeddings
MAX_IMAGES = None         # Limit for quick experiments (None = use all)
RANDOM_SEED = 69

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# Options: "raw_pixels", "handcrafted", "cnn_embeddings"
FEATURE_TYPES = ["raw_pixels", "handcrafted", "cnn_embeddings"]
CNN_BACKBONE = "resnet18"  # For embeddings: resnet18, resnet34, vgg16, efficientnet_b0

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
K_RANGE = (4, 11)         # Range of k to try (e.g. 2 to 10)
K_MEANS_INIT = "k-means++"
K_MEANS_N_INIT = 30
K_MEANS_MAX_ITER = 300

# ---------------------------------------------------------------------------
# PCA (optional, for dimensionality reduction before K-Means)
# ---------------------------------------------------------------------------
USE_PCA_FOR_RAW = True    # Raw pixels are high-dim; PCA can help
PCA_VARIANCE_RETAINED = 0.95
