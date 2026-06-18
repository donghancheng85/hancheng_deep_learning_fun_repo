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
    "lessons/section11_pytorch_model_deployment/models/effnet_b2/"
    "effnet_b2_pizza_steak_sushi_2026-06-14_11-26-23.pth"
)
test_dir = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased/test"
)

# ── Class names (sorted to match DataLoader order) ────────────────────────────────────
CLASS_NAMES = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
print(f"Class names: {CLASS_NAMES}")

# ── Load model ────────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading model from: {MODEL_PATH}")
model, transform = create_effnet_b2_model(num_classes=len(CLASS_NAMES), seed=42)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("[INFO] Model loaded successfully.")

# ── Test image paths ──────────────────────────────────────────────────────────────────
test_data_paths = sorted(test_dir.glob("*/*.jpg"))
print(f"\n[INFO] Found {len(test_data_paths)} test images in: {test_dir}")

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
    # pred_prob is a 1×C tensor; extract the max confidence value
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
fps = 1 / avg_time_s  # sequential single-image FPS
print("\n── Prediction Summary ──")
print(f"  Total images : {len(df)}")
print(f"  Accuracy     : {accuracy:.2f}%")
print(f"  Avg time/img : {avg_time_ms:.2f} ms")
print(f"  FPS          : {fps:.1f}  (sequential, single-image)")
print(
    df.groupby(["ground_truth", "pred_class"])
    .size()
    .rename("count")
    .reset_index()
    .to_string(index=False)
)

# ── Save CSV ──────────────────────────────────────────────────────────────────────────
results_dir = Path("lessons/section11_pytorch_model_deployment/results")
results_dir.mkdir(parents=True, exist_ok=True)
csv_path = results_dir / "effnet_b2_predictions.csv"
df.to_csv(csv_path, index=False)
print(f"\n[INFO] Results saved to: {csv_path}")
