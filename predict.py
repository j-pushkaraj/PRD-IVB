import os
import sys

from PIL import Image, UnidentifiedImageError

import torch
import torch.nn.functional as F
from torchvision import models, transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "good_bad_jaguars_leopards.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def load_model():
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    return model, class_names


def predict_image(img_path: str):
    model, class_names = load_model()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = load_image(img_path)
    if img is None:
        print("Could not open image:", img_path)
        return

    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    print("Classes:", class_names)
    for i, cls in enumerate(class_names):
        print(f"{cls}: {probs[i].item():.3f}")

    pred_idx = probs.argmax().item()
    print("PREDICTION:", class_names[pred_idx].upper())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print("File not found:", img_path)
        sys.exit(1)

    predict_image(img_path)
