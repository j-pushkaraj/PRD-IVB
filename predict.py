# #!/usr/bin/env python3
# """
# predict.py — UPDATED FOR CLIP-LABELED JAGUAR vs LEOPARD MODEL

# Usage:
#   python predict.py path/to/image.jpg
#   python predict.py path/to/folder --out predictions.csv --copy predictions/preview

# Model classes:
#    "good" = Jaguar
#    "bad"  = Leopard
# """

# import os
# import sys
# import argparse
# from pathlib import Path
# from typing import List, Tuple
# import csv
# import shutil

# from PIL import Image, UnidentifiedImageError

# import torch
# import torch.nn.functional as F
# import numpy as np


# # =========================================================
# # DEFAULTS
# # =========================================================
# DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
# DEFAULT_MODEL = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "models",
#     "good_bad_jaguars_leopards.pth"
# )

# SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# LABEL_MAP = {
#     "good": "Jaguar",
#     "bad": "Leopard"
# }


# # =========================================================
# # HELPERS
# # =========================================================
# def is_image_file(p: Path) -> bool:
#     return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


# def find_images(root: Path) -> List[Path]:
#     root = Path(root)
#     if root.is_file():
#         return [root] if is_image_file(root) else []
#     imgs = [p for p in root.rglob("*") if is_image_file(p)]
#     imgs.sort()
#     return imgs


# def load_model(model_path: str, device: str):
#     """Load trained MobileNetV3 model + preprocessing"""
#     from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

#     model_path = Path(model_path)
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found: {model_path}")

#     checkpoint = torch.load(str(model_path), map_location=device)

#     if "model_state" not in checkpoint or "class_names" not in checkpoint:
#         raise RuntimeError("Checkpoint missing keys: 'model_state' and/or 'class_names'")

#     class_names = list(checkpoint["class_names"])

#     weights = MobileNet_V3_Small_Weights.DEFAULT
#     preprocess = weights.transforms()

#     model = mobilenet_v3_small(weights=weights)
#     in_features = model.classifier[-1].in_features
#     model.classifier[-1] = torch.nn.Linear(in_features, len(class_names))

#     model.load_state_dict(checkpoint["model_state"])
#     model = model.to(device)
#     model.eval()

#     return model, class_names, preprocess


# def predict_one(model, preprocess, img_path: Path, device: str) -> np.ndarray:
#     """Returns probabilities array for the image."""
#     try:
#         img = Image.open(img_path).convert("RGB")
#     except Exception:
#         raise RuntimeError(f"Could not open image: {img_path}")

#     x = preprocess(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         logits = model(x)
#         probs = F.softmax(logits, dim=1)[0].cpu().numpy()

#     return probs


# # =========================================================
# # MAIN
# # =========================================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input", help="Path to an image or folder")
#     parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .pth model file")
#     parser.add_argument("--out", default="predictions/predictions.csv", help="CSV output path")
#     parser.add_argument("--copy", default="predictions/preview", help="Copy predicted images into folders")
#     parser.add_argument("--device", default=DEVICE_DEFAULT, help="'cuda' or 'cpu'")
#     parser.add_argument("--topk", type=int, default=2, help="Show top-k predictions")
#     args = parser.parse_args()

#     input_path = Path(args.input)
#     if not input_path.exists():
#         print("ERROR: Input path does not exist:", input_path)
#         sys.exit(2)

#     # Load model
#     try:
#         model, class_names, preprocess = load_model(args.model, args.device)
#     except Exception as e:
#         print("ERROR loading model:", e)
#         sys.exit(3)

#     # Determine images to process
#     imgs = find_images(input_path)
#     if not imgs:
#         print("No images found.")
#         sys.exit(0)

#     # Prepare output directories
#     out_csv = Path(args.out)
#     out_csv.parent.mkdir(parents=True, exist_ok=True)

#     copy_dir = Path(args.copy) if args.copy else None
#     if copy_dir:
#         copy_dir.mkdir(parents=True, exist_ok=True)
#         for cls in class_names:
#             (copy_dir / LABEL_MAP.get(cls, cls)).mkdir(parents=True, exist_ok=True)

#     # Predict
#     results = []
#     for p in imgs:
#         try:
#             probs = predict_one(model, preprocess, p, args.device)
#         except Exception as e:
#             print("Skipping:", p, "Error:", e)
#             continue

#         pred_idx = int(np.argmax(probs))
#         score = float(probs[pred_idx])

#         raw_label = class_names[pred_idx]
#         human_label = LABEL_MAP.get(raw_label, raw_label)

#         print(f"{p.name:25s} → {human_label:8s}  score={score:.3f}")

#         # Copy image into predicted folder
#         if copy_dir:
#             dst_folder = copy_dir / human_label
#             dst = dst_folder / p.name
#             try:
#                 shutil.copy(p, dst)
#             except:
#                 shutil.copy(p, dst_folder / f"{p.stem}_copy{p.suffix}")

#         results.append((str(p), human_label, score))

#     # Write CSV
#     with open(out_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "predicted_label", "score"])
#         writer.writerows(results)

#     print(f"\nCSV saved to: {out_csv}")
#     if copy_dir:
#         print(f"Predicted images copied to: {copy_dir}")


# if __name__ == "__main__":
#     main()












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
