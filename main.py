import os
from typing import List, Tuple, Optional, Dict

from PIL import Image, UnidentifiedImageError

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F


# =========================================================
# GLOBAL CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Put ONLY GOOD (normal) images here
GOOD_DIR = os.path.join(DATA_DIR, "golden_run")

# Where to save augmented GOOD images
AUG_DIR = os.path.join(DATA_DIR, "augmented")
AUG_GOOD_DIR = os.path.join(AUG_DIR, "good")

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(AUG_GOOD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "one_class_good_detector.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# UTILITIES
# =========================================================
def load_image(path: str) -> Optional[Image.Image]:
    """Load an image as RGB, or return None if unreadable."""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def list_files(folder: str) -> List[str]:
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]


# =========================================================
# FEATURE & CLASSIFIER (ResNet50)
# =========================================================
def build_resnet_feature_and_classifier():
    """
    Build:
      - feature extractor (ResNet50 without final FC)
      - classifier (full ResNet50) for semantic labels
    """
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT

    # Feature extractor
    feat_model = resnet50(weights=weights)
    feat_model.fc = nn.Identity()
    feat_model = feat_model.to(DEVICE)
    feat_model.eval()

    # Classifier
    cls_model = resnet50(weights=weights).to(DEVICE)
    cls_model.eval()

    tf = weights.transforms()
    labels = weights.meta["categories"]

    @torch.no_grad()
    def embed_pil(img: Image.Image) -> torch.Tensor:
        x = tf(img).unsqueeze(0).to(DEVICE)
        feat = feat_model(x)
        return feat.squeeze(0).cpu()

    @torch.no_grad()
    def classify_pil(img: Image.Image, k: int = 3):
        x = tf(img).unsqueeze(0).to(DEVICE)
        logits = cls_model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=k)
        result = []
        for i, idx in enumerate(top_idxs):
            label = labels[idx.item()]
            p = top_probs[i].item()
            result.append((label, p))
        return result

    return embed_pil, classify_pil


# =========================================================
# STEP 1: Embed GOOD images + augmentations
# =========================================================
def compute_good_embeddings_and_augment(
    num_augs_per_image: int = 5,
) -> Tuple[torch.Tensor, List[str]]:
    """
    1) Load all GOOD images from GOOD_DIR.
    2) Embed them with ResNet50.
    3) Generate augmentations and embed those too.
    4) Return features and paths.
    """
    good_files = list_files(GOOD_DIR)
    if not good_files:
        raise RuntimeError(
            f"No images found in GOOD_DIR={GOOD_DIR}. "
            "Put ONLY GOOD (normal) images there."
        )

    print(f"Found {len(good_files)} GOOD images in {GOOD_DIR}")

    # Clear old augmentations
    for f in list_files(AUG_GOOD_DIR):
        os.remove(f)

    embed_pil, _ = build_resnet_feature_and_classifier()

    # Augmentation pipeline
    aug_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
    ])

    feats = []
    all_paths = []

    # 1) originals
    for p in good_files:
        img = load_image(p)
        if img is None:
            print("Skipping unreadable image:", p)
            continue
        feat = embed_pil(img)
        feats.append(feat)
        all_paths.append(p)

    print(f"Embedded {len(all_paths)} ORIGINAL good images.")

    # 2) augmentations
    aug_count = 0
    for p in good_files:
        img = load_image(p)
        if img is None:
            continue

        for i in range(num_augs_per_image):
            aug_img = aug_tf(img)

            # Save augmented
            base = os.path.splitext(os.path.basename(p))[0]
            aug_name = f"{base}_aug_{i}.jpg"
            aug_path = os.path.join(AUG_GOOD_DIR, aug_name)
            aug_img.save(aug_path)

            # Embed augmented
            feat = embed_pil(aug_img)
            feats.append(feat)
            all_paths.append(aug_path)
            aug_count += 1

    print(f"Generated and embedded {aug_count} AUGMENTED good images.")
    X = torch.stack(feats)  # (N, D)
    print(f"Total GOOD feature vectors (orig + aug): {X.shape[0]} | Dim: {X.shape[1]}")
    return X, all_paths


# =========================================================
# STEP 2: Learn semantic labels of GOOD images
# =========================================================
def learn_good_semantic_labels(top_k_each: int = 3, max_good_labels: int = 5):
    """
    Run ResNet50 classifier on ORIGINAL GOOD images and
    collect which ImageNet labels are typical for GOOD.

    Returns:
      good_labels: List[str] of most frequent semantic labels for GOOD.
    """
    print("Learning semantic labels for GOOD images...")
    good_files = list_files(GOOD_DIR)
    if not good_files:
        raise RuntimeError("GOOD_DIR is empty while learning semantics.")

    _, classify_pil = build_resnet_feature_and_classifier()

    label_counts: Dict[str, float] = {}

    for p in good_files:
        img = load_image(p)
        if img is None:
            continue
        preds = classify_pil(img, k=top_k_each)
        for label, prob in preds:
            label_counts[label] = label_counts.get(label, 0.0) + prob

    # Sort labels by total score (frequency * probability-ish)
    sorted_labels = sorted(
        label_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    good_labels = [lbl for lbl, _ in sorted_labels[:max_good_labels]]

    print("GOOD semantic labels (from CNN):")
    for lbl, score in sorted_labels[:max_good_labels]:
        print(f"  {lbl} (score={score:.3f})")

    return good_labels


# =========================================================
# STEP 3: Fit one-class anomaly model
# =========================================================
def fit_one_class_model():
    """
    Fit a robust one-class model on GOOD features and semantics.
    """
    # 1) Numeric pattern (features + distances)
    X, paths = compute_good_embeddings_and_augment(num_augs_per_image=5)  # (N, D)

    center = X.mean(dim=0)  # (D,)
    dists = torch.norm(X - center.unsqueeze(0), dim=1)  # (N,)

    median_dist = dists.median()
    abs_dev = torch.abs(dists - median_dist)
    mad = abs_dev.median()
    if mad.item() < 1e-6:
        mad = torch.tensor(1e-3)

    k = 2.5  # stricter than 3.0 since we also have semantics now
    threshold = median_dist + k * mad

    print("\n=== ONE-CLASS MODEL STATS (NUMERIC) ===")
    print(f"Num GOOD samples (orig + aug): {len(paths)}")
    print(f"Median distance               : {median_dist.item():.4f}")
    print(f"MAD (robust spread)           : {mad.item():.4f}")
    print(f"k (scale)                     : {k}")
    print(f"Threshold                     : {threshold.item():.4f}")

    # 2) Semantic pattern (ImageNet labels)
    good_labels = learn_good_semantic_labels()
    print("========================================\n")

    state = {
        "center": center,
        "median_dist": median_dist,
        "mad_dist": mad,
        "threshold": threshold,
        "k": k,
        "backbone": "resnet50",
        "good_labels": good_labels,  # <- NEW: semantic GOOD labels
    }

    torch.save(state, MODEL_PATH)
    print(f"✅ One-class GOOD model saved to: {MODEL_PATH}")


def main():
    fit_one_class_model()


if __name__ == "__main__":
    main()
