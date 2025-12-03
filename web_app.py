import os
from typing import Tuple
from io import BytesIO
import base64

from flask import Flask, render_template, request

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


# =========================================================
# PATHS & DEVICE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "one_class_good_detector.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD ONE-CLASS MODEL
# =========================================================
def load_one_class_model():
    state = torch.load(MODEL_PATH, map_location="cpu")
    center = state["center"]
    thresh = float(state["threshold"])
    median = float(state["median_dist"])
    mad = float(state["mad_dist"])
    k = state.get("k", 2.5)
    good_labels = state.get("good_labels", [])
    return center, thresh, median, mad, k, good_labels


# =========================================================
# BUILD RESNET MODELS
# =========================================================
def build_models():
    weights = ResNet50_Weights.DEFAULT

    feat_model = resnet50(weights=weights)
    feat_model.fc = nn.Identity()
    feat_model.eval().to(DEVICE)

    cls_model = resnet50(weights=weights)
    cls_model.eval().to(DEVICE)

    tf = weights.transforms()
    labels = weights.meta["categories"]

    return feat_model, cls_model, tf, labels


CENTER, THRESH, MEDIAN, MAD, K, GOOD_LABELS = load_one_class_model()
FEAT_MODEL, CLS_MODEL, TF, LABELS = build_models()


# =========================================================
# PREDICTION LOGIC
# =========================================================
@torch.no_grad()
def run_model_on_image(pil_img: Image.Image) -> Tuple[str, str, str]:
    pil_img = pil_img.convert("RGB")

    x = TF(pil_img).unsqueeze(0).to(DEVICE)

    # 1) Distance-based anomaly score
    feat = FEAT_MODEL(x)
    dist = torch.norm(feat.cpu() - CENTER).item()
    anomaly_score = (dist - MEDIAN) / max(MAD, 1e-3)

    # 2) Semantic labels from CNN
    logits = CLS_MODEL(x)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, 3)

    raw_labels = [LABELS[idx.item()] for idx in top_idxs]
    probs_vals = [top_probs[i].item() for i in range(len(top_probs))]

    # --- DEBUG PRINTS TO CONSOLE ---
    print("=== INFERENCE DEBUG ===")
    print(f"Distance: {dist:.4f}, Threshold: {THRESH:.4f}, Median: {MEDIAN:.4f}, MAD: {MAD:.4f}")
    for lbl, p in zip(raw_labels, probs_vals):
        print(f"  label={lbl}, prob={p:.3f}")
    print("GOOD_LABELS (train):", GOOD_LABELS)

    # 3) Hybrid decision: distance + semantics

    # Distance GOOD check
    dist_good = dist <= THRESH

    # Semantic GOOD check: ONLY top-1 label, slightly relaxed prob
    MIN_GOOD_LABEL_PROB = 0.25
    if GOOD_LABELS:
        top1_label = raw_labels[0]
        top1_prob = probs_vals[0]
        sem_good = (top1_label in GOOD_LABELS) and (top1_prob >= MIN_GOOD_LABEL_PROB)
    else:
        sem_good = True  # fallback

    is_good = dist_good and sem_good

    # Build stats text for UI
    stats_text = (
        f"Distance: {dist:.3f}\n"
        f"Threshold: {THRESH:.3f}\n"
        f"Anomaly score (z-like): {anomaly_score:.2f}\n"
        f"Distance-OK: {dist_good}\n"
        f"Semantic-OK (top-1 in GOOD set): {sem_good}\n"
        f"GOOD labels (train): {', '.join(GOOD_LABELS) if GOOD_LABELS else 'N/A'}"
    )

    reasons = [f"{lbl} ({p:.2f})" for lbl, p in zip(raw_labels, probs_vals)]

    if is_good:
        prediction = "GOOD"
        reason_text = (
            "Image matches the learned GOOD pattern.\n\n"
            "CNN top labels:\n - " + "\n - ".join(reasons)
        )
    else:
        prediction = "NOT_GOOD"
        reason_text = (
            "Image deviates from the GOOD pattern.\n\n"
            "Either it's too far in feature space, or its semantic class\n"
            "does not match the GOOD reference class.\n\n"
            "CNN believes it resembles:\n - " + "\n - ".join(reasons)
        )

    print("Final decision:", prediction)
    print("========================\n")

    return prediction, stats_text, reason_text


# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    stats = None
    reason = None
    uploaded_image = None  # base64 data URL for preview

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            try:
                # Read bytes once
                img_bytes = file.read()

                # For model
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                prediction, stats, reason = run_model_on_image(pil_img)

                # For preview in UI (base64)
                encoded = base64.b64encode(img_bytes).decode("utf-8")
                # Assume JPEG/PNG-ish; browser will handle it fine
                uploaded_image = f"data:image/jpeg;base64,{encoded}"
            except Exception as e:
                prediction = "ERROR"
                stats = ""
                reason = f"Failed to process image: {e}"
                uploaded_image = None

    return render_template(
        "index.html",
        prediction=prediction,
        stats=stats,
        reason=reason,
        uploaded_image=uploaded_image,
    )


if __name__ == "__main__":
    app.run(debug=True)
