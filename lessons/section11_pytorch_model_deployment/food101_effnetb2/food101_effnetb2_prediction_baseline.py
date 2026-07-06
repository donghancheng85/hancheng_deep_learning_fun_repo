"""
Prediction baseline for EfficientNet-B2 fine-tuned on Food-101 (101 classes).
Uses the official Food-101 test split (meta/test.txt — 25,250 images).
"""

import pandas as pd
import torch
from pathlib import Path

from lessons.section11_pytorch_model_deployment.model_training.effnet_b2_model_creater import (
    create_effnet_b2_model,
)
from lessons.section11_pytorch_model_deployment.model_inference.model_inference_function import (
    predict_on_images,
)
from common.device import get_best_device, print_device_info

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Paths ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    "lessons/section11_pytorch_model_deployment/models/food101_effnetb2/"
    "effnet_b2_food101_mixup_20ep_2026-07-04_20-17-07.pth"
)
DATA_DIR = Path("lessons/section11_pytorch_model_deployment/data/food-101")
IMAGES_DIR = DATA_DIR / "images"
TEST_SPLIT = DATA_DIR / "meta" / "test.txt"

# ── Class names — sorted alphabetically, matches Food101 DataLoader order ─────────────
CLASS_NAMES = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])
print(f"Classes: {len(CLASS_NAMES)} | e.g. {CLASS_NAMES[:3]} … {CLASS_NAMES[-2:]}")

# ── Load model ────────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading model from: {MODEL_PATH}")
model, transform = create_effnet_b2_model(num_classes=len(CLASS_NAMES), seed=42)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("[INFO] Model loaded successfully.")

# ── Test image paths (from official test.txt split) ───────────────────────────────────
# test.txt format: "class_name/image_id"  (no extension)
test_data_paths = [
    IMAGES_DIR / f"{line.strip()}.jpg"
    for line in TEST_SPLIT.read_text().splitlines()
    if line.strip()
]
print(f"\n[INFO] Test images: {len(test_data_paths):,} (from {TEST_SPLIT})")

# ── Run inference ─────────────────────────────────────────────────────────────────────
predictions = predict_on_images(
    paths=test_data_paths,
    model=model,
    transform=transform,
    class_names=CLASS_NAMES,
    device=device,
)

# ── Build results DataFrame ───────────────────────────────────────────────────────────
rows = []
for pred in predictions:
    max_prob = float(pred["pred_prob"].max().item())
    rows.append(
        {
            "image_path": str(pred["image_path"]),
            "ground_truth": pred["ground_truth"],
            "pred_class": pred["pred_class"],
            "pred_prob": round(max_prob, 4),
            "time_s": round(pred["time_s"], 6),
            "correct": pred["correct"],
        }
    )

df = pd.DataFrame(rows)

# ── Summary ───────────────────────────────────────────────────────────────────────────
accuracy = df["correct"].mean() * 100
avg_time_s = df["time_s"].mean()
avg_time_ms = avg_time_s * 1000
fps = 1 / avg_time_s

print("\n── Prediction Summary ──")
print(f"  Total images : {len(df):,}")
print(f"  Accuracy     : {accuracy:.2f}%")
print(f"  Avg time/img : {avg_time_ms:.2f} ms")
print(f"  FPS          : {fps:.1f}  (sequential, single-image)")

# Per-class accuracy (top-5 worst and best classes)
per_class = (
    df.groupby("ground_truth")["correct"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "correct", "count": "total"})
)
per_class["accuracy"] = per_class["correct"] / per_class["total"] * 100

print("\n── Top-5 Best Classes ──")
print(per_class.nlargest(5, "accuracy")[["accuracy", "total"]].to_string())
print("\n── Top-5 Worst Classes ──")
print(per_class.nsmallest(5, "accuracy")[["accuracy", "total"]].to_string())

# ── Save CSV ──────────────────────────────────────────────────────────────────────────
results_dir = Path(
    "lessons/section11_pytorch_model_deployment/results/food101_effnetb2"
)
results_dir.mkdir(parents=True, exist_ok=True)
csv_path = results_dir / "food101_effnetb2_predictions.csv"
df.to_csv(csv_path, index=False)
print(f"\n[INFO] Results saved to: {csv_path}")
