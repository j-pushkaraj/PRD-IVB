import os
import random
import shutil
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
import torch.nn.functional as F


# =========================================================
# GLOBAL CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# All jaguar + leopard images (mixed) go here
MIXED_DIR = os.path.join(DATA_DIR, "golden_run")

AUTO_LABEL_DIR = os.path.join(DATA_DIR, "auto_labels")
AUG_DIR = os.path.join(DATA_DIR, "augmented")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "good_bad_jaguars_leopards.pth")

for d in [AUTO_LABEL_DIR, AUG_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# UTILITIES
# =========================================================
def load_image(path: str):
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
# STEP 1: Deep feature extractor (MobileNetV3)
# =========================================================
def build_feature_extractor():
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    weights = MobileNet_V3_Small_Weights.DEFAULT
    backbone = mobilenet_v3_small(weights=weights)
    backbone.classifier = nn.Identity()  # drop final classifier -> pure features
    backbone = backbone.to(DEVICE)
    backbone.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    @torch.no_grad()
    def embed(path: str) -> torch.Tensor | None:
        img = load_image(path)
        if img is None:
            return None
        x = tf(img).unsqueeze(0).to(DEVICE)
        feat = backbone(x)  # (1, D)
        return feat.squeeze(0).cpu()  # (D,)

    return embed


def compute_embeddings(image_paths: List[str]) -> torch.Tensor:
    """
    Returns tensor of shape (N, D) for all images.
    Skips unreadable images.
    """
    embed = build_feature_extractor()
    feats = []
    valid_paths = []
    for p in image_paths:
        f = embed(p)
        if f is not None:
            feats.append(f)
            valid_paths.append(p)

    if not feats:
        raise RuntimeError("No valid images to embed.")
    X = torch.stack(feats)  # (N, D)
    return X, valid_paths


# =========================================================
# STEP 2: Robust Deep Fuzzy K-means (RDFKM-like)
# =========================================================
class RDFKM:
    """
    Simplified Robust Deep Fuzzy K-means:
      - Deep features from MobileNetV3
      - Fuzzy C-means clustering with robust sample weights
    """

    def __init__(
        self,
        n_clusters: int = 2,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        robust_delta: float = 1.0,
    ):
        self.c = n_clusters
        self.m = m  # fuzzifier
        self.max_iter = max_iter
        self.tol = tol
        self.robust_delta = robust_delta

        self.centers: torch.Tensor | None = None  # (c, D)
        self.U: torch.Tensor | None = None        # (N, c)

    def _init_centers(self, X: torch.Tensor):
        # Simple k-means++ style init
        N, D = X.shape
        centers = torch.empty((self.c, D), dtype=X.dtype)
        # pick first center randomly
        idx0 = torch.randint(0, N, (1,)).item()
        centers[0] = X[idx0]
        # pick others far away
        for k in range(1, self.c):
            dists = torch.min(torch.cdist(X, centers[:k]), dim=1)[0]
            probs = dists / dists.sum()
            idx = torch.multinomial(probs, 1).item()
            centers[k] = X[idx]
        self.centers = centers

    def fit(self, X: torch.Tensor):
        """
        X: (N, D) feature tensor (CPU).
        """
        X = X.clone()
        N, D = X.shape

        if self.centers is None:
            self._init_centers(X)

        # Initialize membership matrix U randomly
        U = torch.rand((N, self.c))
        U = U / U.sum(dim=1, keepdim=True)

        m = self.m
        eps = 1e-8

        for it in range(self.max_iter):
            # --- update centers (robust, with sample weights) ---
            # distances (N, c)
            dists = torch.cdist(X, self.centers) + eps

            # membership with exponent m
            Um = U ** m  # (N, c)

            # robust weights per sample based on closest center
            min_dist, _ = torch.min(dists, dim=1)  # (N,)
            w = 1.0 / (1.0 + (min_dist / self.robust_delta) ** 2)  # (N,)
            w = w.unsqueeze(1)  # (N, 1)

            # weighted Um
            WUm = w * Um  # (N, c)

            centers_new = torch.empty_like(self.centers)
            for k in range(self.c):
                num = (WUm[:, k:k+1] * X).sum(dim=0)
                den = WUm[:, k].sum() + eps
                centers_new[k] = num / den

            center_shift = torch.norm(centers_new - self.centers).item()
            self.centers = centers_new

            # --- update memberships U ---
            dists = torch.cdist(X, self.centers) + eps  # (N, c)
            # standard fuzzy c-means membership update
            # u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
            inv_d = dists ** (-2.0 / (m - 1.0))
            U = inv_d / inv_d.sum(dim=1, keepdim=True)

            if center_shift < self.tol:
                break

        self.U = U

    def hard_labels(self) -> torch.Tensor:
        """
        Returns argmax cluster index per sample (N,)
        """
        if self.U is None:
            raise RuntimeError("RDFKM not fitted yet.")
        return torch.argmax(self.U, dim=1)

    def memberships(self) -> torch.Tensor:
        if self.U is None:
            raise RuntimeError("RDFKM not fitted yet.")
        return self.U


# =========================================================
# STEP 3: Map clusters to Jaguar (GOOD) / Leopard (BAD)
# =========================================================
def build_imagenet_classifier():
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(DEVICE)
    model.eval()
    tf = weights.transforms()
    labels = weights.meta["categories"]

    @torch.no_grad()
    def classify(path: str, topk: int = 5):
        img = load_image(path)
        if img is None:
            return None
        x = tf(img).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=topk)
        return [(labels[idx.item()], top_probs[i].item()) for i, idx in enumerate(top_idxs)]

    return classify


def auto_label_with_rdfkm():
    """
    1) Compute deep features for all images in MIXED_DIR
    2) Run RDFKM (fuzzy clustering) with c=2
    3) For each cluster, look at high-membership images.
       Use ResNet50 to determine which cluster is Jaguar vs Leopard.
    4) Create auto_labels/good (Jaguar) and auto_labels/bad (Leopard)
    """
    print("=== STEP 1 & 2: RDFKM clustering on deep features ===")
    all_images = list_files(MIXED_DIR)
    if not all_images:
        raise RuntimeError(
            f"No images found in {MIXED_DIR}. Put jaguar + leopard images there first."
        )

    X, valid_paths = compute_embeddings(all_images)  # (N, D)
    N = X.shape[0]
    print(f"Embedded {N} images into deep feature space.")

    rdfkm = RDFKM(n_clusters=2, m=2.0, max_iter=100, tol=1e-4, robust_delta=1.0)
    rdfkm.fit(X)
    U = rdfkm.memberships()  # (N, 2)
    hard = rdfkm.hard_labels()  # (N,)

    # Decide mapping of cluster -> (Jaguar/Leopard) via ResNet
    classify = build_imagenet_classifier()

    cluster_info = {}
    for k in range(2):
        cluster_info[k] = {"jaguar_votes": 0, "leopard_votes": 0, "samples": []}

    # Use only images with membership > 0.6 as "confident" examples
    for idx, path in enumerate(valid_paths):
        k = hard[idx].item()
        conf = U[idx, k].item()
        if conf < 0.6:
            continue
        preds = classify(path, topk=5)
        if preds is None:
            continue
        # vote based on keywords in top-5
        jag_vote = 0
        leo_vote = 0
        for label, prob in preds:
            ll = label.lower()
            if "jaguar" in ll:
                jag_vote += prob
            if "leopard" in ll:
                leo_vote += prob
        cluster_info[k]["jaguar_votes"] += jag_vote
        cluster_info[k]["leopard_votes"] += leo_vote
        cluster_info[k]["samples"].append(path)

    print("\nCluster semantic votes (from ResNet50):")
    for k, info in cluster_info.items():
        print(
            f"Cluster {k}: jaguar_votes={info['jaguar_votes']:.3f}, "
            f"leopard_votes={info['leopard_votes']:.3f}, "
            f"samples={len(info['samples'])}"
        )

    # Decide GOOD/BAD mapping
    # Cluster with higher jaguar_votes -> GOOD, the other -> BAD
    jag_scores = {
        k: info["jaguar_votes"] for k, info in cluster_info.items()
    }
    good_cluster = max(jag_scores, key=jag_scores.get)
    bad_cluster = 1 - good_cluster

    print(f"\nGOOD cluster (Jaguar) = {good_cluster}, BAD cluster (Leopard) = {bad_cluster}")

    # Create auto_labels folders
    GOOD_DIR = os.path.join(AUTO_LABEL_DIR, "good")
    BAD_DIR = os.path.join(AUTO_LABEL_DIR, "bad")
    os.makedirs(GOOD_DIR, exist_ok=True)
    os.makedirs(BAD_DIR, exist_ok=True)

    # Clear previous auto-labels
    for folder in [GOOD_DIR, BAD_DIR]:
        for f in list_files(folder):
            os.remove(f)

    # Assign images based on hard labels + cluster mapping
    print("\nAssigning images to GOOD/BAD using RDFKM clusters:")
    for idx, path in enumerate(valid_paths):
        k = hard[idx].item()
        fname = os.path.basename(path)
        if k == good_cluster:
            dst = os.path.join(GOOD_DIR, fname)
            label_str = "GOOD"
        else:
            dst = os.path.join(BAD_DIR, fname)
            label_str = "BAD"
        shutil.copy(path, dst)
        print(f"{fname:20s} -> {label_str} (cluster={k}, conf={U[idx, k].item():.3f})")

    print("\nGOOD_DIR contents:", os.listdir(GOOD_DIR))
    print("BAD_DIR contents:", os.listdir(BAD_DIR))

    if len(os.listdir(GOOD_DIR)) == 0:
        raise RuntimeError("No GOOD images after RDFKM labeling.")
    if len(os.listdir(BAD_DIR)) == 0:
        raise RuntimeError("No BAD images after RDFKM labeling.")

    print("=== RDFKM-based auto-labeling DONE ===\n")


# =========================================================
# STEP 4: Augment GOOD/BAD classes
# =========================================================
def augment_classes(target_count: int = 15):
    print(f"=== STEP 3: Augment GOOD/BAD to {target_count} images per class ===")
    GOOD_SRC = os.path.join(AUTO_LABEL_DIR, "good")
    BAD_SRC = os.path.join(AUTO_LABEL_DIR, "bad")

    AUG_GOOD_DIR = os.path.join(AUG_DIR, "good")
    AUG_BAD_DIR = os.path.join(AUG_DIR, "bad")
    os.makedirs(AUG_GOOD_DIR, exist_ok=True)
    os.makedirs(AUG_BAD_DIR, exist_ok=True)

    # Clear previous augmented data
    for folder in [AUG_GOOD_DIR, AUG_BAD_DIR]:
        for f in list_files(folder):
            os.remove(f)

    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.2),
    ])

    def generate_for_class(src_dir: str, out_dir: str, prefix: str):
        files = list_files(src_dir)
        if not files:
            print("No files in", src_dir)
            return

        count = 0
        # copy originals
        for path in files:
            img = load_image(path)
            if img is None:
                continue
            img.save(os.path.join(out_dir, f"{prefix}_orig_{count}.jpg"))
            count += 1

        # augment until target_count
        while count < target_count:
            src = random.choice(files)
            img = load_image(src)
            if img is None:
                continue
            aug_img = augment(img)
            aug_img.save(os.path.join(out_dir, f"{prefix}_aug_{count}.jpg"))
            count += 1

        print(f"{out_dir}: {count} images")

    generate_for_class(GOOD_SRC, AUG_GOOD_DIR, prefix="good")
    generate_for_class(BAD_SRC, AUG_BAD_DIR, prefix="bad")
    print("=== STEP 3 DONE ===\n")


# =========================================================
# STEP 5: Train GOOD vs BAD classifier
# =========================================================
def train_classifier(epochs: int = 10, val_ratio: float = 0.2):
    print("=== STEP 4: Train GOOD/BAD classifier (jaguar vs leopard) ===")

    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(AUG_DIR, transform=train_tf)
    if len(dataset) < 2:
        raise RuntimeError("Not enough images in augmented dataset to train.")

    class_names = dataset.classes  # ['bad', 'good']
    print("Classes:", class_names)

    # Train/val split
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return correct / total if total > 0 else 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        val_acc = evaluate(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.3f}  ValAcc={val_acc:.3f}"
        )

    torch.save(
        {"model_state": model.state_dict(), "class_names": class_names},
        MODEL_PATH,
    )

    print("Model saved to:", MODEL_PATH)
    print("=== STEP 4 DONE ===\n")


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    # 1 & 2) RDFKM clustering + semantic mapping to GOOD/BAD
    auto_label_with_rdfkm()

    # 3) Data augmentation
    augment_classes(target_count=15)

    # 4) Train GOOD vs BAD classifier
    train_classifier(epochs=10, val_ratio=0.2)


if __name__ == "__main__":
    main()
