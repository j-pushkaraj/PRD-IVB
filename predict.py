#!/usr/bin/env python3
"""
predict.py

Usage:
    # Single image
    python predict.py path/to/img.jpg

    # Folder (recursive)
    python predict.py path/to/folder --out results.csv --save_vis predictions/

Options:
    --model PATH   Path to .pth model file
    --topk N       Show top-k results (default 2)
    --device D     'cpu' or 'cuda' (auto if not given)
    --save_vis DIR Copy images to DIR/<class>/ for visual inspection
"""

import os
import sys
import argparse
import csv
import shutil
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms


# -----------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "jaguar_leopard_mnet_large.pth")


# -----------------------------------------------------------
# IMAGE UTILS
# -----------------------------------------------------------
def valid_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED


def find_images(path: Path) -> List[Path]:
    """Recursively find images."""
    if path.is_file():
        return [path] if valid_img(path) else []

    imgs = [p for p in path.rglob("*") if valid_img(p)]
    imgs.sort()
    return imgs


def load_img(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
def load_model(model_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    if "model_state" not in ckpt or "class_names" not in ckpt:
        raise RuntimeError("Checkpoint missing keys ['model_state', 'class_names']")

    class_names = ckpt["class_names"]

    # Load weight transforms
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()

    # Build model
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])

    model = model.to(device)
    model.eval()

    return model, preprocess, class_names


# -----------------------------------------------------------
# PREDICT ONE IMAGE
# -----------------------------------------------------------
def predict_one(model, preprocess, img_path: Path, device: str):
    img = load_img(img_path)
    if img is None:
        raise RuntimeError(f"Cannot open image: {img_path}")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    return probs


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image or folder path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .pth file")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--out", default="predictions.csv", help="CSV output")
    parser.add_argument("--save_vis", default="", help="Copy predictions into folders")
    args = parser.parse_args()

    device = args.device
    input_path = Path(args.input)

    if not input_path.exists():
        print("Input does not exist:", input_path)
        sys.exit(1)

    # Load model
    try:
        model, preprocess, class_names = load_model(args.model, device)
    except Exception as e:
        print("Model load failed:", e)
        sys.exit(2)

    # Gather images
    images = find_images(input_path)
    if len(images) == 0:
        print("No valid images found.")
        sys.exit(0)

    # Visual output folders
    vis_root = Path(args.save_vis)
    if vis_root != Path(""):
        for c in class_names:
            (vis_root / c).mkdir(parents=True, exist_ok=True)

    # CSV setup
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [["image", "pred_label", "score"]]

    print("\n=== PREDICTING ===\n")

    for img_path in images:
        try:
            probs = predict_one(model, preprocess, img_path, device)
        except Exception as e:
            print("Skipping:", img_path, "error:", e)
            continue

        # Top prediction
        pred_idx = int(probs.argmax())
        pred_label = class_names[pred_idx]
        pred_score = float(probs[pred_idx])

        # Print
        print(f"{img_path.name:25s} => {pred_label} ({pred_score:.3f})")

        rows.append([str(img_path), pred_label, pred_score])

        # Copy to prediction folder
        if vis_root != Path(""):
            dst = vis_root / pred_label / img_path.name
            try:
                shutil.copy(img_path, dst)
            except:
                pass

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print("\nSaved CSV:", out_csv)
    if vis_root != Path(""):
        print("Saved visual predictions to:", vis_root)


if __name__ == "__main__":
    main()
