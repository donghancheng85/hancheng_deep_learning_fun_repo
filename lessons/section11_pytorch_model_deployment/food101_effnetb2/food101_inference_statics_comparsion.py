"""
Inference statistics for EfficientNet-B2 fine-tuned on Food-101 (101 classes).

Covers the same five industry-standard dimensions as inference_statics_comparsion.py:
  1. Overall accuracy, speed, ECE summary
  2. Per-class metrics  (precision / recall / F1 — top/bottom 10 shown)
  3. Confusion analysis (top-20 misclassification pairs + per-class accuracy heatmap)
  4. Confidence calibration (reliability diagram + ECE)
  5. Confidence distribution (correct vs wrong histogram)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(
    "lessons/section11_pytorch_model_deployment/results/food101_effnetb2"
)
MODEL_PATH = Path(
    "lessons/section11_pytorch_model_deployment/models/food101_effnetb2/"
    "effnet_b2_food101_mixup_20ep_2026-07-04_20-17-07.pth"
)
N_BINS = 10

# ── Load results ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_DIR / "food101_effnetb2_predictions.csv")
CLASS_NAMES = sorted(df["ground_truth"].unique())
print(f"[INFO] Loaded {len(df):,} predictions across {len(CLASS_NAMES)} classes.")


# ── ECE helper ────────────────────────────────────────────────────────────────────────
def expected_calibration_error(
    confidences: pd.Series, correct_flags: pd.Series, n_bins: int = 10
) -> float:
    """ECE = Σ_k (n_k / N) * |acc_k − conf_k| across confidence bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    if total == 0:
        return 0.0
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (
            (confidences >= lo) & (confidences <= hi)
            if i == n_bins - 1
            else (confidences >= lo) & (confidences < hi)
        )
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        ece += (n_k / total) * abs(
            float(correct_flags[mask].mean()) - float(confidences[mask].mean())
        )
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────────────
# 1. OVERALL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. OVERALL SUMMARY")
print("=" * 60)

accuracy = df["correct"].mean() * 100
avg_time_s = df["time_s"].mean()
avg_ms = avg_time_s * 1000
fps = 1 / avg_time_s
size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
ece = expected_calibration_error(df["pred_prob"], df["correct"], n_bins=N_BINS)

summary = pd.DataFrame(
    [
        {
            "Model": "EfficientNet-B2 Food-101",
            "Test images": len(df),
            "Accuracy (%)": round(accuracy, 2),
            "Avg latency (ms)": round(avg_ms, 2),
            "FPS (sequential)": round(fps, 1),
            "Model size (MB)": round(size_mb, 2),
            "ECE (10 bins)": round(ece, 4),
        }
    ]
).set_index("Model")
print(summary.to_string())

# ─────────────────────────────────────────────────────────────────────────────────────
# 2. PER-CLASS METRICS  — top / bottom 10 by F1
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PER-CLASS METRICS (precision / recall / F1)")
print("=" * 60)

report_dict = classification_report(
    df["ground_truth"],
    df["pred_class"],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
assert isinstance(report_dict, dict)
per_class_df = (
    pd.DataFrame(report_dict)
    .T.loc[CLASS_NAMES, ["precision", "recall", "f1-score", "support"]]
    .astype({"precision": float, "recall": float, "f1-score": float, "support": int})
)

print(
    f"\nMacro avg  — F1: {report_dict['macro avg']['f1-score']:.4f} | "
    f"P: {report_dict['macro avg']['precision']:.4f} | "
    f"R: {report_dict['macro avg']['recall']:.4f}"
)
print(f"Weighted avg — F1: {report_dict['weighted avg']['f1-score']:.4f}")

print("\n── Top-10 classes by F1 ──")
print(per_class_df.nlargest(10, "f1-score").to_string())
print("\n── Bottom-10 classes by F1 ──")
print(per_class_df.nsmallest(10, "f1-score").to_string())

# ─────────────────────────────────────────────────────────────────────────────────────
# 3A. TOP-20 MISCLASSIFICATION PAIRS
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3A. TOP-20 MISCLASSIFICATION PAIRS")
print("=" * 60)

wrong = df[~df["correct"]]
misclass = (
    wrong.groupby(["ground_truth", "pred_class"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(20)
)
print(misclass.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────────────
# 3B. PER-CLASS ACCURACY HEATMAP  (10 × 11 grid)
# ─────────────────────────────────────────────────────────────────────────────────────
per_class_acc_flat: np.ndarray = np.array(
    df.groupby("ground_truth")["correct"].mean().reindex(CLASS_NAMES).values,
    dtype=float,
)
# Pad to the next multiple of 10 with NaN so the grid is rectangular
COLS = 10
pad = (COLS - len(per_class_acc_flat) % COLS) % COLS
per_class_acc_padded = np.concatenate([per_class_acc_flat, np.full(pad, np.nan)])
per_class_acc = per_class_acc_padded.reshape(-1, COLS)  # rows × 10 cols

n_rows, n_cols = per_class_acc.shape
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(per_class_acc, cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, label="Accuracy")
ax.set_xticks(range(n_cols))
ax.set_yticks(range(n_rows))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title(
    "Per-class Accuracy Heatmap — EfficientNet-B2 on Food-101\n"
    "(green = high, red = low; classes sorted alphabetically)",
    fontsize=12,
)

# Annotate each cell with class name + accuracy
for idx, cls in enumerate(CLASS_NAMES):
    row, col = divmod(idx, COLS)
    acc_val = per_class_acc[row, col]
    ax.text(
        col,
        row,
        f"{cls}\n{acc_val:.0%}",
        ha="center",
        va="center",
        fontsize=6,
        color="black",
    )

plt.tight_layout()
plt.savefig(RESULTS_DIR / "per_class_accuracy_heatmap.png", dpi=150)
print(
    "\n[INFO] Per-class accuracy heatmap → results/food101_effnetb2/per_class_accuracy_heatmap.png"
)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# 4. CONFIDENCE CALIBRATION  (reliability diagram + ECE)
# ─────────────────────────────────────────────────────────────────────────────────────
bins = np.linspace(0, 1, N_BINS + 1)
bin_accs, bin_confs, bin_counts = [], [], []
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    mask = (
        (df["pred_prob"] >= lo) & (df["pred_prob"] <= hi)
        if i == N_BINS - 1
        else (df["pred_prob"] >= lo) & (df["pred_prob"] < hi)
    )
    if mask.sum() == 0:
        continue
    bin_accs.append(float(df["correct"][mask].mean()))
    bin_confs.append(float(df["pred_prob"][mask].mean()))
    bin_counts.append(int(mask.sum()))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.bar(bin_confs, bin_accs, width=1 / N_BINS * 0.8, alpha=0.6, label="Accuracy per bin")
ax.plot(bin_confs, bin_accs, "o-", color="tab:orange")
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Actual accuracy")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f"Reliability Diagram — EfficientNet-B2 Food-101\nECE = {ece:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "calibration_diagram.png", dpi=150)
print(f"[INFO] Calibration diagram saved | ECE = {ece:.4f}")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# 5. CONFIDENCE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(
    df.loc[df["correct"], "pred_prob"],
    bins=30,
    alpha=0.7,
    label=f"Correct ({df['correct'].sum():,})",
    color="steelblue",
)
ax.hist(
    df.loc[~df["correct"], "pred_prob"],
    bins=30,
    alpha=0.7,
    label=f"Wrong ({(~df['correct']).sum():,})",
    color="tomato",
)
ax.set_xlabel("Predicted confidence")
ax.set_ylabel("Count")
ax.set_title("Confidence Distribution — EfficientNet-B2 Food-101")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confidence_distribution.png", dpi=150)
print("[INFO] Confidence distribution saved.")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# 6. F1 SCORE DISTRIBUTION  (histogram across 101 classes)
# ─────────────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(per_class_df["f1-score"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(
    per_class_df["f1-score"].mean(),
    color="tomato",
    linestyle="--",
    label=f"Mean F1 = {per_class_df['f1-score'].mean():.3f}",
)
ax.set_xlabel("F1 score")
ax.set_ylabel("Number of classes")
ax.set_title("F1 Score Distribution across 101 Food Classes")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "f1_distribution.png", dpi=150)
print("[INFO] F1 distribution saved.")
plt.show()

print("\n[DONE] All statistics complete.")
print(f"       Plots saved to: {RESULTS_DIR}")
