"""
Temperature Scaling for EfficientNet-B2 Food-101
──────────────────────────────────────────────────
What is temperature scaling?
────────────────────────────
A neural network outputs raw scores ("logits") z_i for each class.
Normally: p_i = softmax(z_i) = exp(z_i) / Σ exp(z_j)

Temperature scaling divides all logits by a single scalar T before softmax:
  p_i(T) = exp(z_i / T) / Σ exp(z_j / T)

  T > 1  →  probabilities get softer  (fixes overconfidence)
  T < 1  →  probabilities get sharper (fixes underconfidence)
  T = 1  →  no change (uncalibrated baseline)

The optimal T is found by minimising Negative Log-Likelihood (NLL) on a small
calibration split.  Temperature scaling NEVER changes accuracy — it only shifts
confidence values to better match the actual accuracy of each confidence band.

Why does this matter?
──────────────────────
Our model had ECE = 0.221, meaning its mid-range confidence predictions are
systematically too high (~0.65 confidence where actual accuracy is ~0.85).
After calibration, ECE typically drops to < 0.05 and the reliability diagram
lies on the perfect-calibration diagonal.

Pipeline:
  1. Reserve 20% of test images as calibration set (no overlap with training).
  2. Collect raw logits by running the model in eval mode (no softmax).
  3. Optimise T on calibration set using scipy.
  4. Report before/after ECE on the remaining 80% held-out set.
  5. Save T to JSON for reuse in production inference.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from lessons.section11_pytorch_model_deployment.model_training.effnet_b2_model_creater import (
    create_effnet_b2_model,
)
from common.device import get_best_device, print_device_info

# ── Config ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    "lessons/section11_pytorch_model_deployment/models/food101_effnetb2/"
    "effnet_b2_food101_mixup_20ep_2026-07-04_20-17-07.pth"
)
DATA_DIR = Path("lessons/section11_pytorch_model_deployment/data/food-101")
IMAGES_DIR = DATA_DIR / "images"
TEST_SPLIT = DATA_DIR / "meta" / "test.txt"
RESULTS_DIR = Path(
    "lessons/section11_pytorch_model_deployment/results/food101_effnetb2"
)
TEMP_PATH = RESULTS_DIR / "optimal_temperature.json"

CALIB_FRAC = 0.20  # 20% of test set for calibration (~5,050 images)
BATCH_SIZE = 64
N_BINS = 10

device = get_best_device()
print_device_info(device)

# ── Class names ───────────────────────────────────────────────────────────────────────
CLASS_NAMES = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ── Load model ────────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading model from {MODEL_PATH}")
model, transform = create_effnet_b2_model(num_classes=len(CLASS_NAMES), seed=42)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()
print("[INFO] Model loaded.")


# ── Dataset helper ────────────────────────────────────────────────────────────────────
class PathDataset(Dataset):
    """Minimal dataset: list of image paths → (tensor, label_idx)."""

    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = CLASS_TO_IDX[path.parent.name]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Collect logits ────────────────────────────────────────────────────────────────────
def collect_logits(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run batch inference and return (logits [N×101], labels [N]) on CPU."""
    loader = DataLoader(
        PathDataset(paths, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for imgs, labels in tqdm(loader, desc="Collecting logits"):
            logits = model(imgs.to(device))  # raw logits, no softmax
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ── ECE ───────────────────────────────────────────────────────────────────────────────
def compute_ece(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = N_BINS
) -> float:
    """ECE from full probability vectors (not just max prob)."""
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].float().mean().item()
        conf = confidences[mask].mean().item()
        ece += (mask.sum().item() / len(probs)) * abs(acc - conf)
    return ece


# ── NLL objective for scipy ───────────────────────────────────────────────────────────
def nll_with_temperature(T: float, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Negative log-likelihood of logits/T evaluated on labels."""
    scaled_probs = F.softmax(logits / T, dim=1)
    nll = F.nll_loss(scaled_probs.log(), labels).item()
    return nll


# ── Split paths into calibration / held-out ───────────────────────────────────────────
all_lines = [l.strip() for l in TEST_SPLIT.read_text().splitlines() if l.strip()]
# Use deterministic split (first CALIB_FRAC fraction) so results are reproducible
n_calib = int(len(all_lines) * CALIB_FRAC)
calib_paths = [IMAGES_DIR / f"{l}.jpg" for l in all_lines[:n_calib]]
heldout_paths = [IMAGES_DIR / f"{l}.jpg" for l in all_lines[n_calib:]]

print(f"\n[INFO] Calibration set : {len(calib_paths):,} images")
print(f"[INFO] Held-out set    : {len(heldout_paths):,} images")

# ── Collect logits for both splits ────────────────────────────────────────────────────
print("\n[INFO] Collecting calibration logits …")
calib_logits, calib_labels = collect_logits(calib_paths)

print("\n[INFO] Collecting held-out logits …")
heldout_logits, heldout_labels = collect_logits(heldout_paths)

# ── Baseline ECE (T = 1) ──────────────────────────────────────────────────────────────
baseline_probs = F.softmax(heldout_logits, dim=1)
baseline_ece = compute_ece(baseline_probs, heldout_labels)
baseline_acc = (
    baseline_probs.argmax(dim=1).eq(heldout_labels).float().mean().item() * 100
)
print(f"\n── Before calibration ──────────────────────")
print(f"  T           = 1.000  (no scaling)")
print(f"  ECE         = {baseline_ece:.4f}")
print(f"  Accuracy    = {baseline_acc:.2f}%  (unchanged by T)")

# ── Optimise T on calibration set ─────────────────────────────────────────────────────
print("\n[INFO] Optimising temperature on calibration set …")
result = minimize_scalar(
    nll_with_temperature,
    args=(calib_logits, calib_labels),
    bounds=(0.1, 10.0),
    method="bounded",
)
T_opt = float(result.x)
print(f"[INFO] Optimal T = {T_opt:.4f}  (NLL = {result.fun:.4f})")

# ── Calibrated ECE (T = T_opt) ────────────────────────────────────────────────────────
calibrated_probs = F.softmax(heldout_logits / T_opt, dim=1)
calibrated_ece = compute_ece(calibrated_probs, heldout_labels)
calibrated_acc = (
    calibrated_probs.argmax(dim=1).eq(heldout_labels).float().mean().item() * 100
)

print(f"\n── After calibration ───────────────────────")
print(f"  T           = {T_opt:.4f}")
print(f"  ECE         = {calibrated_ece:.4f}  (was {baseline_ece:.4f})")
print(
    f"  Accuracy    = {calibrated_acc:.2f}%  (was {baseline_acc:.2f}% — should be identical)"
)

ece_reduction = (baseline_ece - calibrated_ece) / baseline_ece * 100
print(f"\n  ECE reduced by {ece_reduction:.1f}%")

# ── Save T ────────────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(TEMP_PATH, "w") as f:
    json.dump(
        {
            "temperature": T_opt,
            "baseline_ece": baseline_ece,
            "calibrated_ece": calibrated_ece,
        },
        f,
        indent=2,
    )
print(f"\n[INFO] Optimal T saved → {TEMP_PATH}")
print("       Load it with: T = json.load(open(TEMP_PATH))['temperature']")
print("       Then apply  : probs = torch.softmax(logits / T, dim=1)")
