"""
Feature extraction for plant images.
Supports: raw pixels, handcrafted (color + HOG), CNN embeddings.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from skimage import feature as skfeature
    from skimage import color as skcolor
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Raw pixel features
# ---------------------------------------------------------------------------

def load_images_from_paths(paths: list, size=(64, 64)) -> np.ndarray:
    """Load images from paths into (N, H, W, 3) array, resized and [0,1] normalized."""
    imgs = []
    for p in tqdm(paths, desc="Loading images", leave=False):
        img = Image.open(p).convert("RGB")
        img = img.resize((size[1], size[0]))
        arr = np.array(img).astype(np.float32) / 255.0
        imgs.append(arr)
    return np.array(imgs)


def extract_raw_pixels(images: np.ndarray, size=(64, 64)) -> np.ndarray:
    """
    Extract raw pixel features: resize, flatten.
    images: (N, H, W, 3) or list of arrays
    Returns: (N, D) where D = size[0]*size[1]*3
    """
    if isinstance(images, list):
        images = np.array(images)
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    feats = []
    for i in range(len(images)):
        img = Image.fromarray((images[i] * 255).astype(np.uint8) if images[i].max() <= 1 else images[i].astype(np.uint8))
        img = img.resize((size[1], size[0]))
        arr = np.array(img).astype(np.float32) / 255.0
        feats.append(arr.flatten())
    return np.vstack(feats)


# ---------------------------------------------------------------------------
# Handcrafted features (color histogram + HOG)
# ---------------------------------------------------------------------------

def extract_handcrafted(images: np.ndarray, size=(64, 64), bins=32) -> np.ndarray:
    """
    Handcrafted features: color histogram (RGB) + HOG.
    images: (N, H, W, 3)
    Returns: (N, D)
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for handcrafted features: pip install scikit-image")

    if isinstance(images, list):
        images = np.array(images)
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    feats = []
    for i in range(len(images)):
        img = images[i]
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

        # Resize
        pil = Image.fromarray(img)
        pil = pil.resize((size[1], size[0]))
        img = np.array(pil)

        # Color histogram per channel (bins per channel, 3 channels)
        hist_r = np.histogram(img[..., 0], bins=bins, range=(0, 256))[0].astype(np.float32)
        hist_g = np.histogram(img[..., 1], bins=bins, range=(0, 256))[0].astype(np.float32)
        hist_b = np.histogram(img[..., 2], bins=bins, range=(0, 256))[0].astype(np.float32)
        color_feat = np.concatenate([hist_r, hist_g, hist_b])

        # HOG (grayscale)
        gray = skcolor.rgb2gray(img)
        hog_feat = skfeature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

        feats.append(np.concatenate([color_feat, hog_feat]))
    return np.vstack(feats)


def extract_raw_pixels_from_paths(paths: list, size=(64, 64)) -> np.ndarray:
    """Extract raw pixel features from image paths."""
    images = load_images_from_paths(paths, size)
    return extract_raw_pixels(images, size)


def extract_handcrafted_from_paths(paths: list, size=(64, 64), bins=32) -> np.ndarray:
    """Extract handcrafted features from image paths."""
    images = load_images_from_paths(paths, size)
    return extract_handcrafted(images, size, bins)


# ---------------------------------------------------------------------------
# CNN embeddings (pretrained, no final layer)
# ---------------------------------------------------------------------------

def _build_cnn_extractor(backbone="resnet18"):
    """Build pretrained CNN without classifier head. Returns (B, C, 1, 1) or (B, D)."""
    # Compatibility: torchvision 0.13+ uses Weights enum; older uses string
    try:
        w = models.ResNet18_Weights.IMAGENET1K_V1
    except AttributeError:
        w = "IMAGENET1K_V1"

    if backbone == "resnet18":
        model = models.resnet18(weights=w)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif backbone == "resnet34":
        try:
            w34 = models.ResNet34_Weights.IMAGENET1K_V1
        except AttributeError:
            w34 = "IMAGENET1K_V1"
        model = models.resnet34(weights=w34)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif backbone == "vgg16":
        try:
            wv = models.VGG16_Weights.IMAGENET1K_V1
        except AttributeError:
            wv = "IMAGENET1K_V1"
        model = models.vgg16(weights=wv)
        model = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten(),
                                    *list(model.classifier.children())[:-1])
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model.eval()


def extract_cnn_embeddings(image_paths: list, backbone="resnet18", size=(224, 224), batch_size=32, device=None):
    """
    Extract CNN embeddings from image paths.
    Returns: (N, D) numpy array
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision required for CNN embeddings")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    extractor = _build_cnn_extractor(backbone).to(device)

    from PIL import Image
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            out = extractor(batch)
        out = out.flatten(1) if out.dim() > 2 else out
        embeddings.append(out.cpu().numpy())
    return np.vstack(embeddings)
