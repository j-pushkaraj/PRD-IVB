import os
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image
import gradio as gr


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
# RESNET FEATURE + SEMANTIC MODEL
# =========================================================
def build_models():
    from torchvision.models import resnet50, ResNet50_Weights

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
def predict_image(img: Image.Image) -> Tuple[str, str, str]:
    img = img.convert("RGB")

    x = TF(img).unsqueeze(0).to(DEVICE)

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

    reasons = [f"{lbl} ({p:.2f})" for lbl, p in zip(raw_labels, probs_vals)]

    # Semantic GOOD check
    MIN_GOOD_LABEL_PROB = 0.30
    if GOOD_LABELS:
        sem_good = any(
            (lbl in GOOD_LABELS) and (p >= MIN_GOOD_LABEL_PROB)
            for lbl, p in zip(raw_labels, probs_vals)
        )
    else:
        sem_good = True  # fallback

    # Distance GOOD check
    dist_good = dist <= THRESH

    # Final hybrid decision
    is_good = dist_good and sem_good

    stats_text = (
        f"Distance: {dist:.3f}\n"
        f"Threshold: {THRESH:.3f}\n"
        f"Anomaly score: {anomaly_score:.2f}\n"
        f"Distance-OK: {dist_good}\n"
        f"Semantic-OK: {sem_good}\n"
        f"GOOD labels (train): {', '.join(GOOD_LABELS) if GOOD_LABELS else 'N/A'}"
    )

    if is_good:
        reason_text = (
            "✅ Image matches the learned GOOD pattern.\n\n"
            "CNN top labels:\n - " + "\n - ".join(reasons)
        )
        return "✅ GOOD", stats_text, reason_text

    # NOT GOOD
    reason_text = (
        "❌ Image deviates from the GOOD pattern.\n\n"
        "Either it's too far in feature space, or its semantic class\n"
        "does not match the GOOD reference class.\n\n"
        "CNN believes it resembles:\n - " + "\n - ".join(reasons)
    )

    return "❌ NOT_GOOD", stats_text, reason_text


# =========================================================
# GRADIO UI (compatible with older versions)
# =========================================================
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Stats"),
        gr.Textbox(label="Reason (CNN explanation)"),
    ],
    title="Visual Anomaly Detector for Manufacturing",
    description=(
        "Upload an image.\n"
        "Model was trained ONLY on GOOD images and uses a hybrid of\n"
        "deep features + CNN semantics to decide if it's GOOD or NOT_GOOD."
    ),
)

if __name__ == "__main__":
    demo.launch()
