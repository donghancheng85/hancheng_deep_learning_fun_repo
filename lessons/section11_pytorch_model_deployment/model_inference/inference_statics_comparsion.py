"""
Model comparison: EfficientNet-B2 vs ViT-B/16
─────────────────────────────────────────────
In the ML industry, models are typically compared across five dimensions:

  1. Accuracy / Error metrics   — overall and per-class (precision, recall, F1)
  2. Confusion matrix           — where each model mis-classifies
  3. Confidence calibration     — does high probability actually mean high accuracy?
  4. Speed / throughput         — avg latency and sequential FPS
  5. Model size                 — disk footprint (trade-off for deployment)

This script covers all five, using the pre-computed inference CSVs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def expected_calibration_error(
    confidences: pd.Series, correct_flags: pd.Series, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_k (n_k / N) * |acc_k - conf_k| across confidence bins.
    Lower is better; 0 means perfectly calibrated confidence estimates.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    if total == 0:
        return 0.0

    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_k = int(mask.sum())
        if n_k == 0:
            continue

        acc_k = float(correct_flags[mask].mean())
        conf_k = float(confidences[mask].mean())
        ece += (n_k / total) * abs(acc_k - conf_k)

    return float(ece)

# ── Load results ──────────────────────────────────────────────────────────────────────
results_dir = Path("lessons/section11_pytorch_model_deployment/results")
df_eff = pd.read_csv(results_dir / "effnet_b2_predictions.csv")
df_vit = pd.read_csv(results_dir / "vit_b16_predictions.csv")

CLASS_NAMES = sorted(df_eff["ground_truth"].unique())
MODELS = {"EfficientNet-B2": df_eff, "ViT-B/16": df_vit}

MODEL_PATHS = {
    "EfficientNet-B2": Path(
        "lessons/section11_pytorch_model_deployment/models/effnet_b2/"
        "effnet_b2_pizza_steak_sushi_2026-06-14_11-26-23.pth"
    ),
    "ViT-B/16": Path(
        "lessons/section11_pytorch_model_deployment/models/vit_b16/"
        "vit_b16_pizza_steak_sushi_2026-06-14_12-03-43.pth"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────────────
# 1. OVERALL ACCURACY & SPEED SUMMARY
# ─────────────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. OVERALL SUMMARY")
print("=" * 60)
summary_rows = []
for name, df in MODELS.items():
    acc = df["correct"].mean() * 100
    avg_ms = df["time_s"].mean() * 1000
    fps = 1 / df["time_s"].mean()
    size_mb = MODEL_PATHS[name].stat().st_size / (1024 * 1024)
    ece = expected_calibration_error(df["pred_prob"], df["correct"], n_bins=10)
    summary_rows.append(
        {
            "Model": name,
            "Accuracy (%)": round(acc, 2),
            "Avg latency (ms)": round(avg_ms, 2),
            "FPS (sequential)": round(fps, 1),
            "Model size (MB)": round(size_mb, 2),
            "ECE (10 bins)": round(ece, 4),
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("Model")
print(summary_df.to_string())

# ─────────────────────────────────────────────────────────────────────────────────────
# 2. PER-CLASS METRICS  (precision / recall / F1)
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PER-CLASS METRICS")
print("=" * 60)
for name, df in MODELS.items():
    print(f"\n── {name} ──")
    print(
        classification_report(
            df["ground_truth"], df["pred_class"], target_names=CLASS_NAMES
        )
    )

# ─────────────────────────────────────────────────────────────────────────────────────
# 3. CONFUSION MATRICES  (side-by-side plot)
# ─────────────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, df) in zip(axes, MODELS.items()):
    cm = confusion_matrix(df["ground_truth"], df["pred_class"], labels=CLASS_NAMES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)

plt.tight_layout()
plt.savefig(results_dir / "confusion_matrices.png", dpi=150)
print("\n[INFO] Confusion matrix plot saved → results/confusion_matrices.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# 4. CONFIDENCE CALIBRATION  (reliability diagram)
#    Bucket predictions by confidence; plot mean confidence vs actual accuracy per bin.
#    A perfectly calibrated model falls on the diagonal y = x.
# ─────────────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Confidence Calibration (Reliability Diagram)", fontsize=14, fontweight="bold"
)

N_BINS = 10
bins = np.linspace(0, 1, N_BINS + 1)

for ax, (name, df) in zip(axes, MODELS.items()):
    bin_accs, bin_confs, bin_counts = [], [], []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == N_BINS - 1:
            mask = (df["pred_prob"] >= lo) & (df["pred_prob"] <= hi)
        else:
            mask = (df["pred_prob"] >= lo) & (df["pred_prob"] < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(float(df["correct"][mask].mean()))
        bin_confs.append(float(df["pred_prob"][mask].mean()))
        bin_counts.append(mask.sum())

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(
        bin_confs,
        bin_accs,
        width=1 / N_BINS * 0.8,
        alpha=0.6,
        label="Accuracy per bin",
    )
    ax.plot(bin_confs, bin_accs, "o-", color="tab:orange")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(name)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(results_dir / "calibration_diagram.png", dpi=150)
print("[INFO] Calibration diagram saved → results/calibration_diagram.png")
plt.show()

print("\n" + "=" * 60)
print("4B. CALIBRATION ERROR (ECE)")
print("=" * 60)
for name, df in MODELS.items():
    ece = expected_calibration_error(df["pred_prob"], df["correct"], n_bins=N_BINS)
    print(f"{name:16s} ECE ({N_BINS} bins): {ece:.4f}")

# ─────────────────────────────────────────────────────────────────────────────────────
# 5. CONFIDENCE DISTRIBUTION  (histogram per model)
#    High-confidence, high-accuracy models have mass near 1.0.
# ─────────────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Prediction Confidence Distribution", fontsize=14, fontweight="bold")

for ax, (name, df) in zip(axes, MODELS.items()):
    ax.hist(
        df.loc[df["correct"], "pred_prob"],
        bins=20,
        alpha=0.7,
        label="Correct",
        color="steelblue",
    )
    ax.hist(
        df.loc[~df["correct"], "pred_prob"],
        bins=20,
        alpha=0.7,
        label="Wrong",
        color="tomato",
    )
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.savefig(results_dir / "confidence_distribution.png", dpi=150)
print("[INFO] Confidence distribution saved → results/confidence_distribution.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# 6. DISAGREEMENT ANALYSIS
#    Images where the two models give different predictions — useful for error auditing.
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. DISAGREEMENT ANALYSIS")
print("=" * 60)

merged = df_eff[
    ["image_path", "ground_truth", "pred_class", "correct", "pred_prob"]
].merge(
    df_vit[["image_path", "pred_class", "correct", "pred_prob"]],
    on="image_path",
    suffixes=("_eff", "_vit"),
)

disagree = merged[merged["pred_class_eff"] != merged["pred_class_vit"]].copy()
print(f"Images where models disagree : {len(disagree)} / {len(merged)}")

# Further split: only-eff-correct, only-vit-correct, both wrong
only_eff = disagree[disagree["correct_eff"] & ~disagree["correct_vit"]]
only_vit = disagree[~disagree["correct_eff"] & disagree["correct_vit"]]
both_wrong = disagree[~disagree["correct_eff"] & ~disagree["correct_vit"]]

print(f"  EfficientNet-B2 correct, ViT wrong : {len(only_eff)}")
print(f"  ViT-B/16 correct, EfficientNet wrong: {len(only_vit)}")
print(f"  Both wrong                          : {len(both_wrong)}")

# ─────────────────────────────────────────────────────────────────────────────────────
# 7. SPEED vs ACCURACY  (scatter plot — the key deployment trade-off)
# ─────────────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, df in MODELS.items():
    acc = df["correct"].mean() * 100
    avg_ms = df["time_s"].mean() * 1000
    size_mb = MODEL_PATHS[name].stat().st_size / (1024 * 1024)
    ax.scatter(
        avg_ms, acc, s=size_mb * 2, alpha=0.8, label=f"{name} ({size_mb:.1f} MB)"
    )
    ax.annotate(
        name, (avg_ms, acc), textcoords="offset points", xytext=(8, 4), fontsize=9
    )

ax.set_xlabel("Avg inference latency (ms)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Speed vs Accuracy\n(bubble size ∝ model size on disk)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(results_dir / "speed_vs_accuracy.png", dpi=150)
print("[INFO] Speed-vs-accuracy plot saved → results/speed_vs_accuracy.png")
plt.show()

print("\n[DONE] All comparisons complete.")
