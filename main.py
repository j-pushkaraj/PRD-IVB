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
