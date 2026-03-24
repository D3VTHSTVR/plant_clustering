"""
Data loading and preprocessing for plant images.
Supports: local ImageFolder, flat folder, and iNaturalist 2021.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# iNaturalist 2021
# ---------------------------------------------------------------------------

def load_inaturalist2021_plants(
    root: Path,
    version="2021_train_mini",
    max_images=None,
    download=True,
    random_state=42,
):
    """
    Load plant images from iNaturalist 2021.
    Filters for kingdom Plantae only.
    Returns: (paths, labels, label_to_idx)
    - paths: list of image file paths
    - labels: optional species/class labels (for evaluation)
    - label_to_idx: optional mapping (e.g. genus -> idx)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for iNaturalist. pip install torch torchvision")

    try:
        from torchvision.datasets import INaturalist
    except ImportError:
        raise ImportError(
            "torchvision.datasets.INaturalist requires torchvision >= 0.12. "
            "Upgrade: pip install --upgrade torchvision"
        )

    dataset = INaturalist(
        root=str(root),
        version=version,
        target_type=["full", "kingdom"],  # species id + kingdom id
        download=download,
    )

    # iNaturalist 2021 dir names: "00000_Plantae_Tracheophyta_..."
    # Build set of plant category ids (kingdom = second field)
    plant_cat_ids = set()
    for cat_id, dir_name in enumerate(dataset.all_categories):
        parts = dir_name.split("_")
        if len(parts) >= 2 and parts[1] == "Plantae":
            plant_cat_ids.add(cat_id)

    paths = []
    labels = []
    for cat_id, fname in dataset.index:
        if cat_id in plant_cat_ids:
            dir_name = dataset.all_categories[cat_id]
            image_path = os.path.join(dataset.root, dir_name, fname)
            paths.append(image_path)
            labels.append(cat_id)

    # Random subsample for diverse species (dataset ordered by species)
    np.random.seed(random_state)
    if max_images and len(paths) > max_images:
        idx = np.random.choice(len(paths), max_images, replace=False)
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx]

    labels = np.array(labels) if labels else None
    unique = sorted(set(labels)) if labels is not None else []
    label_to_idx = {str(i): i for i in unique} if unique else None

    return paths, labels, label_to_idx


def load_dataset(source="local", **kwargs):
    """
    Load image paths and labels based on dataset source.
    source: "local" | "inaturalist2021"
    Returns: (paths, labels, label_to_idx)
    """
    if source == "inaturalist2021":
        return load_inaturalist2021_plants(**kwargs)
    return load_all_image_paths(**kwargs)


# ---------------------------------------------------------------------------
# Image loading (for unsupervised K-Means pipeline)
# ---------------------------------------------------------------------------

def load_all_image_paths(folder: Path, extensions=(".jpg", ".jpeg", ".png"), max_images=None):
    """
    Load all image paths for K-Means pipeline. Supports:
    - ImageFolder: folder/class_name/img.jpg (returns paths, labels, label_to_idx)
    - Flat: folder/img.jpg (returns paths, None, None)
    Labels are optional (for evaluation only); clustering is unsupervised.
    """
    paths = []
    labels_list = []
    label_to_idx = {}
    idx = 0
    n = 0

    for item in sorted(folder.iterdir()):
        if max_images and n >= max_images:
            break
        if item.is_dir():
            class_name = item.name
            if class_name not in label_to_idx:
                label_to_idx[class_name] = idx
                idx += 1
            for f in sorted(item.iterdir()):
                if max_images and n >= max_images:
                    break
                if f.suffix.lower() in extensions:
                    paths.append(str(f))
                    labels_list.append(label_to_idx[class_name])
                    n += 1
        elif item.suffix.lower() in extensions:
            paths.append(str(item))
            labels_list.append(-1)  # No label
            n += 1

    labels = np.array(labels_list) if labels_list and any(l >= 0 for l in labels_list) else None
    return paths, labels, label_to_idx if label_to_idx else None


def load_images_from_folder(folder: Path, extensions=(".jpg", ".jpeg", ".png")):
    """
    Load image paths and labels from an ImageFolder-style directory.
    Expects: folder/class_name/image.jpg
    Returns: (paths, labels), label_to_idx mapping
    """
    paths = []
    labels = []
    label_to_idx = {}
    idx = 0

    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in label_to_idx:
            label_to_idx[class_name] = idx
            idx += 1
        for f in class_dir.iterdir():
            if f.suffix.lower() in extensions:
                paths.append(str(f))
                labels.append(label_to_idx[class_name])

    return paths, np.array(labels), label_to_idx


def load_image_as_array(path: str, size=(224, 224)):
    """Load a single image as numpy array, resized and normalized to [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class PlantDataset(Dataset):
        """PyTorch Dataset for plant images."""

        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]
            if isinstance(label, (list, np.ndarray)):
                label = int(label[0]) if len(label) > 0 else 0
            else:
                label = int(label)
            return img, label


    def get_transforms(img_size=(224, 224), augment_train=True):
        """Build train and val transforms."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if augment_train:
            train_tf = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_tf = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                normalize
            ])
        val_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
        return train_tf, val_tf


    def create_dataloaders(
        raw_data_dir: Path,
        batch_size=32,
        img_size=(224, 224),
        val_split=0.2,
        num_workers=0,
        random_state=42
    ):
        """
        Create train and validation DataLoaders from ImageFolder layout.
        """
        paths, labels, label_to_idx = load_images_from_folder(raw_data_dir)
        if len(paths) == 0:
            raise FileNotFoundError(
                f"No images found in {raw_data_dir}. "
                f"Expected subfolders like: trees/, bushes/, flowers/ with images inside."
            )

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=val_split, stratify=labels, random_state=random_state
        )

        train_tf, val_tf = get_transforms(img_size)
        train_ds = PlantDataset(train_paths, train_labels, transform=train_tf)
        val_ds = PlantDataset(val_paths, val_labels, transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False
        )

        return train_loader, val_loader, label_to_idx
