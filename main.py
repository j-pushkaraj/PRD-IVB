# #!/usr/bin/env python3
# """
# FINAL VERSION — BEST PIPELINE
# Uses CLIP for auto-labeling Jaguar vs Leopard
# No clustering, no ImageNet voting.
# Much more accurate for small datasets.
# """

# import os
# import random
# import shutil
# from typing import List, Optional
# from pathlib import Path
# import uuid

# from PIL import Image, UnidentifiedImageError

# import torch
# import torch.nn.functional as F
# from torch import nn, optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, datasets

# try:
#     import clip
# except ImportError:
#     print("ERROR: CLIP not installed. Install it with: pip install openai-clip")
#     raise
# import numpy as np

# # =========================================================
# # GLOBAL CONFIG
# # =========================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "data")

# # All mixed jaguar + leopard images go here
# MIXED_DIR = os.path.join(DATA_DIR, "golden_run1")

# AUTO_LABEL_DIR = os.path.join(DATA_DIR, "auto_labels")
# AUG_DIR = os.path.join(DATA_DIR, "augmented")

# MODELS_DIR = os.path.join(BASE_DIR, "models")
# MODEL_PATH = os.path.join(MODELS_DIR, "good_bad_jaguars_leopards.pth")

# for d in [AUTO_LABEL_DIR, AUG_DIR, MODELS_DIR]:
#     os.makedirs(d, exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)


# # =========================================================
# # UTILITIES
# # =========================================================
# def load_image(path: str) -> Optional[Image.Image]:
#     try:
#         return Image.open(path).convert("RGB")
#     except (UnidentifiedImageError, OSError):
#         return None


# def list_images(folder: str) -> List[str]:
#     """Recursively list image files in folder."""
#     exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#     p = Path(folder)
#     if not p.exists():
#         return []
#     return [str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]


# # =========================================================
# # STEP 1 — CLIP AUTO-LABELING (BEST METHOD)
# # =========================================================
# def auto_label_with_clip():
#     print("=== AUTO-LABEL USING CLIP (Jaguar vs Leopard) ===")

#     all_images = list_images(MIXED_DIR)
#     if not all_images:
#         raise RuntimeError("No images found in golden_run1/")

#     model, preprocess = clip.load("ViT-B/32", device=DEVICE)

#     # Text prompts for CLIP
#     text_tokens = clip.tokenize([
#         "a photo of a jaguar, large cat with rosettes",
#         "a photo of a leopard, large cat with circular spots"
#     ]).to(DEVICE)

#     with torch.no_grad():
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#     GOOD_DIR = os.path.join(AUTO_LABEL_DIR, "good")  # Jaguar
#     BAD_DIR = os.path.join(AUTO_LABEL_DIR, "bad")    # Leopard

#     os.makedirs(GOOD_DIR, exist_ok=True)
#     os.makedirs(BAD_DIR, exist_ok=True)

#     # Clear previous labels
#     for folder in [GOOD_DIR, BAD_DIR]:
#         for p in list_images(folder):
#             os.remove(p)

#     for img_path in all_images:
#         img = load_image(img_path)
#         if img is None:
#             continue

#         image_input = preprocess(img).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             img_feat = model.encode_image(image_input)
#             img_feat /= img_feat.norm(dim=-1, keepdim=True)
#             similarity = (img_feat @ text_features.T).softmax(dim=-1)[0]

#         jag_prob = similarity[0].item()
#         lep_prob = similarity[1].item()

#         fname = os.path.basename(img_path)
#         unique = f"{uuid.uuid4().hex[:8]}_{fname}"

#         if jag_prob > lep_prob:
#             dst = os.path.join(GOOD_DIR, unique)
#             label = "JAGUAR"
#         else:
#             dst = os.path.join(BAD_DIR, unique)
#             label = "LEOPARD"

#         shutil.copy(img_path, dst)
#         print(f"{fname:20s}  Jaguar={jag_prob:.3f}  Leopard={lep_prob:.3f}  --> {label}")

#     print("\nCLIP auto-labeling done.\n")


# # =========================================================
# # STEP 2 — AUGMENTATION
# # =========================================================
# def augment_classes(target_count: int = 20):
#     print(f"=== STEP 2: Augmenting to {target_count} images per class ===")

#     GOOD_SRC = os.path.join(AUTO_LABEL_DIR, "good")
#     BAD_SRC = os.path.join(AUTO_LABEL_DIR, "bad")

#     AUG_GOOD = os.path.join(AUG_DIR, "good")
#     AUG_BAD = os.path.join(AUG_DIR, "bad")
#     os.makedirs(AUG_GOOD, exist_ok=True)
#     os.makedirs(AUG_BAD, exist_ok=True)

#     for d in [AUG_GOOD, AUG_BAD]:
#         for f in list_images(d):
#             os.remove(f)

#     augment = transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#         transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
#         transforms.RandomHorizontalFlip(),
#     ])

#     def process_class(src_dir, out_dir, prefix):
#         files = list_images(src_dir)
#         if not files:
#             print("No images in", src_dir)
#             return

#         count = 0

#         # Copy originals
#         for path in files:
#             img = load_image(path)
#             if img is None:
#                 continue
#             unique = f"{prefix}_orig_{uuid.uuid4().hex[:6]}.jpg"
#             img.resize((224, 224)).save(os.path.join(out_dir, unique))
#             count += 1

#         # Augment to target_count
#         while count < target_count:
#             path = random.choice(files)
#             img = load_image(path)
#             if img is None:
#                 continue
#             aug_img = augment(img)
#             if isinstance(aug_img, torch.Tensor):
#                 aug_img = transforms.ToPILImage()(aug_img)
#             aug_img = aug_img.resize((224, 224))
#             unique = f"{prefix}_aug_{uuid.uuid4().hex[:6]}.jpg"
#             aug_img.save(os.path.join(out_dir, unique))
#             count += 1

#         print(f"{out_dir}: {count} images")

#     process_class(GOOD_SRC, AUG_GOOD, "good")
#     process_class(BAD_SRC, AUG_BAD, "bad")

#     print("=== Augmentation DONE ===\n")


# # =========================================================
# # STEP 3 — TRAIN CLASSIFIER
# # =========================================================
# def train_classifier(epochs: int = 12, val_ratio: float = 0.2):
#     print("=== STEP 3: Training classifier ===")

#     from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
#     weights = MobileNet_V3_Small_Weights.DEFAULT
#     preprocess = weights.transforms()

#     dataset = datasets.ImageFolder(AUG_DIR, transform=preprocess)
#     class_names = dataset.classes
#     print("Classes =", class_names)

#     val_size = max(1, int(len(dataset) * val_ratio))
#     train_size = len(dataset) - val_size

#     train_ds, val_ds = random_split(
#         dataset, [train_size, val_size],
#         generator=torch.Generator().manual_seed(SEED)
#     )

#     train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

#     model = mobilenet_v3_small(weights=weights)
#     model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
#     model = model.to(DEVICE)

#     opt = optim.Adam(model.parameters(), lr=1e-4)
#     loss_fn = nn.CrossEntropyLoss()

#     def eval_acc(loader):
#         model.eval()
#         correct = total = 0
#         with torch.no_grad():
#             for x, y in loader:
#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 preds = model(x).argmax(1)
#                 correct += (preds == y).sum().item()
#                 total += y.size(0)
#         return correct / total if total else 0

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0
#         correct = total = 0

#         for x, y in train_loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             opt.zero_grad()
#             out = model(x)
#             loss = loss_fn(out, y)
#             loss.backward()
#             opt.step()

#             running_loss += loss.item() * x.size(0)
#             correct += (out.argmax(1) == y).sum().item()
#             total += y.size(0)

#         train_loss = running_loss / total
#         train_acc = correct / total
#         val_acc = eval_acc(val_loader)

#         print(f"Epoch {epoch+1}/{epochs}  TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.3f}  ValAcc={val_acc:.3f}")

#     torch.save({"model_state": model.state_dict(), "class_names": class_names}, MODEL_PATH)
#     print("\nModel saved to", MODEL_PATH)


# # =========================================================
# # MAIN
# # =========================================================
# def main():
#     auto_label_with_clip()
#     augment_classes(target_count=20)
#     train_classifier(epochs=12, val_ratio=0.2)


# if __name__ == "__main__":
#     main()














#!/usr/bin/env python3
"""
FULL HYBRID PIPELINE:
CLIP Auto-label (Jaguar vs Leopard),
CNN Clustering Fusion,
Augmentation,
Training (MobileNetV3-LARGE),
Final Model Export.
"""

import os
import uuid
import random
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

import clip
from sklearn.cluster import KMeans


# =========================================================
# GLOBAL CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MIXED_DIR = os.path.join(DATA_DIR, "golden_run1")    # your folder
AUTO_LABEL_DIR = os.path.join(DATA_DIR, "auto_labels")
AUG_DIR = os.path.join(DATA_DIR, "augmented")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "jaguar_leopard_mnet_large.pth")

# create folders
for d in [AUTO_LABEL_DIR, AUG_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# UTILITIES
# =========================================================

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def load_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def list_images(folder: str) -> List[str]:
    folder = Path(folder)
    return [str(f) for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]


# =========================================================
# STEP 1 — HYBRID AUTO-LABELING (CLIP + CNN)
# =========================================================

def auto_label_hybrid(margin=0.02):

    print("\n=== STEP 1: HYBRID AUTO-LABELING (CLIP + CNN) ===\n")

    image_paths = list_images(MIXED_DIR)
    if len(image_paths) == 0:
        raise RuntimeError("No images found in data/golden_run1 !")

    # ----------------- Load CLIP -----------------
    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    # ----------------- Prompt Ensemble -----------------
    jag_prompts = [
        "a jaguar with rosette-shaped spots",
        "a big wild jaguar",
        "a jaguar in the rainforest",
        "a muscular jaguar cat",
        "a jaguar with yellow fur and rosettes"
    ]

    lep_prompts = [
        "a leopard with circular spots",
        "a big wild leopard",
        "a leopard standing on a rock",
        "a leopard with golden fur",
        "a leopard resting on a tree branch"
    ]

    all_prompts = jag_prompts + lep_prompts
    tokens = clip.tokenize(all_prompts).to(DEVICE)

    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    jag_feats = text_feats[: len(jag_prompts)]
    lep_feats = text_feats[len(jag_prompts):]

    # ------------------ CNN Model (for clustering) ------------------
    from torchvision.models import resnet50, ResNet50_Weights
    cnn_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
    cnn_model.fc = nn.Identity()
    cnn_model.eval()
    cnn_preprocess = ResNet50_Weights.DEFAULT.transforms()

    # ------------------ Output folders ------------------
    GOOD = os.path.join(AUTO_LABEL_DIR, "good")      # jaguar
    BAD = os.path.join(AUTO_LABEL_DIR, "bad")        # leopard
    UNC = os.path.join(AUTO_LABEL_DIR, "uncertain")

    for d in [GOOD, BAD, UNC]:
        os.makedirs(d, exist_ok=True)
        for f in Path(d).glob("*"):
            f.unlink()

    # =========================================================
    # COMPUTE CNN FEATURES
    # =========================================================
    print("\nExtracting CNN features for clustering...")

    cnn_feats = []
    clean_img_paths = []

    for p in image_paths:
        img = load_image(p)
        if img is None:
            continue

        x = cnn_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = cnn_model(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        cnn_feats.append(feat.cpu())
        clean_img_paths.append(p)

    cnn_feats = torch.cat(cnn_feats, dim=0)

    # ------------------ K-Means Clustering ------------------
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_ids = kmeans.fit_predict(cnn_feats)

    # ------------------ Determine cluster identity via CLIP ------------------
    print("\nDetermining cluster identity using CLIP votes...")

    cluster_votes = {0: 0, 1: 0}

    for p, cid in zip(clean_img_paths, cluster_ids):
        img = load_image(p)
        if img is None:
            continue

        x = clip_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)

        jag_s = float((f @ jag_feats.T).mean())
        lep_s = float((f @ lep_feats.T).mean())

        if jag_s > lep_s:
            cluster_votes[cid] += 1
        else:
            cluster_votes[cid] -= 1

    jag_cluster = 0 if cluster_votes[0] > cluster_votes[1] else 1
    lep_cluster = 1 - jag_cluster

    print(f"Jaguar Cluster = {jag_cluster}")
    print(f"Leopard Cluster = {lep_cluster}")

    # =========================================================
    # FINAL LABELING USING HYBRID FUSION
    # =========================================================

    print("\nLabeling images...\n")

    for p, cid in zip(clean_img_paths, cluster_ids):

        img = load_image(p)
        if img is None:
            continue

        # CLIP scoring
        x = clip_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)

        jag_s = float((f @ jag_feats.T).mean())
        lep_s = float((f @ lep_feats.T).mean())

        # CNN cluster prediction
        cnn_pred = "jaguar" if cid == jag_cluster else "leopard"

        # Hybrid fusion decision
        fusion_score = 0.6 * (jag_s - lep_s) + 0.4 * (1 if cnn_pred == "jaguar" else -1)

        pred = "jaguar" if fusion_score > 0 else "leopard"
        confident = abs(fusion_score) >= margin

        out_dir = UNC if not confident else (GOOD if pred == "jaguar" else BAD)

        fname = os.path.basename(p)
        unique = f"{uuid.uuid4().hex[:6]}_{fname}"
        shutil.copy(p, os.path.join(out_dir, unique))

        print(f"{fname:20s}  ->  {pred.upper():7s}  ({'CONF' if confident else 'UNCERT'})")

    print("\nHybrid labeling complete.\n")


# =========================================================
# STEP 2 — DATA AUGMENTATION
# =========================================================

def augment_classes(target_count=20):

    print("\n=== STEP 2: DATA AUGMENTATION ===")

    from torchvision.transforms import ColorJitter

    GOOD_SRC = os.path.join(AUTO_LABEL_DIR, "good")
    BAD_SRC = os.path.join(AUTO_LABEL_DIR, "bad")

    GOOD_OUT = os.path.join(AUG_DIR, "good")
    BAD_OUT = os.path.join(AUG_DIR, "bad")

    for d in [GOOD_OUT, BAD_OUT]:
        os.makedirs(d, exist_ok=True)
        for f in Path(d).glob("*"):
            f.unlink()

    aug_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    def augment_folder(src, dst, prefix):
        files = list_images(src)
        if len(files) == 0:
            print(f"WARNING: No images in {src}")
            return

        count = 0

        # copy original
        for f in files:
            img = load_image(f)
            if img is None:
                continue
            out = os.path.join(dst, f"{prefix}_orig_{uuid.uuid4().hex[:6]}.jpg")
            img.resize((224, 224)).save(out)
            count += 1

        # augment more
        while count < target_count:
            f = random.choice(files)
            img = load_image(f)
            if img is None:
                continue

            aug = aug_tf(img)
            if isinstance(aug, torch.Tensor):
                aug = transforms.ToPILImage()(aug)

            out = os.path.join(dst, f"{prefix}_aug_{uuid.uuid4().hex[:6]}.jpg")
            aug.resize((224, 224)).save(out)
            count += 1

        print(f"{dst} -> {count} images")

    augment_folder(GOOD_SRC, GOOD_OUT, "good")
    augment_folder(BAD_SRC, BAD_OUT, "bad")

    print("\nAugmentation done.\n")


# =========================================================
# STEP 3 — TRAIN CLASSIFIER (MobileNetV3-LARGE)
# =========================================================

def train_classifier(epochs=12, val_ratio=0.2):

    print("=== STEP 3: TRAIN MobileNetV3-LARGE CLASSIFIER ===")

    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()

    dataset = datasets.ImageFolder(AUG_DIR, transform=preprocess)
    class_names = dataset.classes
    print("Classes:", class_names)

    if len(class_names) != 2:
        raise RuntimeError("Expected 2 classes (good & bad).")

    # train-val split
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

    # Model
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
        return correct / total

    # train loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        train_acc = evaluate(train_loader)
        val_acc = evaluate(val_loader)
        print(f"Epoch {epoch}/{epochs} Loss={total_loss/train_size:.4f} TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f}")

    # Save final model
    torch.save({"model_state": model.state_dict(), "class_names": class_names}, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)
    print("\nTraining complete.\n")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n============================================")
    print("        HYBRID BIG CAT CLASSIFIER v1")
    print("============================================")

    auto_label_hybrid(margin=0.02)
    augment_classes(target_count=25)
    train_classifier(epochs=12)

    print("=== PIPELINE FINISHED SUCCESSFULLY ===\n")


if __name__ == "__main__":
    main()
