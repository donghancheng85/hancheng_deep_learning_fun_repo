"""
b_pytorch20_intro_1_5_epoch_run_compare.py

Compares total train+test time of a baseline ResNet50 vs a torch.compile'd
ResNet50 on the SVHN dataset.

Key insight:
  - torch.compile() is lazy: JIT compilation fires on the first forward pass,
    so epoch-1 of the compiled model is slower (compile overhead).
  - From epoch-2 onward the compiled model runs faster kernel-fused code.
  - Total-time comparison captures both the cost and the benefit.
"""

import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
from pathlib import Path

# --- project imports ---
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn
from going_modular.pytorch_project.engine import train
from lessons.section12_pytorch20_intro.common_functions import (
    create_model,
    create_compile_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCHS = 5
BATCH_SIZE = 64
NUM_CLASSES = 10
IMG_SIZE = 224  # ResNet50 canonical input size
DATA_DIR = Path("lessons/section12_pytorch20_intro/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

device = get_best_device()
print_device_info(device)

# Enable TF32 for matmul on Ampere+ GPUs (RTX 30xx / 40xx / 50xx).
# cuDNN conv TF32 is already True by default; this gates the matmul path.
# 'high' → TF32 (10-bit mantissa, ~8× faster).  Negligible accuracy impact
# for image classification.  The compiled model benefits most from this.
torch.set_float32_matmul_precision("high")
print(f"float32 matmul precision: {torch.get_float32_matmul_precision()}")

# ---------------------------------------------------------------------------
# Data — SVHN (32×32 RGB) upscaled to 224×224 for ResNet50
# ---------------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],  # SVHN channel stats
            std=[0.1980, 0.2010, 0.1970],
        ),
    ]
)

print("\nLoading SVHN dataset ...")
train_dataset = datasets.SVHN(
    root=DATA_DIR, split="train", download=True, transform=transform
)
test_dataset = datasets.SVHN(
    root=DATA_DIR, split="test", download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

# ---------------------------------------------------------------------------
# Model info (torchinfo summary on the baseline model — CPU is fine here)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ResNet50 Model Summary")
print("=" * 60)
_info_model = create_model(num_classes=NUM_CLASSES)
summary(
    _info_model,
    input_size=(1, 3, IMG_SIZE, IMG_SIZE),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    depth=2,
    verbose=1,
)
del _info_model  # free memory before training

# ---------------------------------------------------------------------------
# 1. Baseline ResNet50 (no compile)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("1 / 2  —  Baseline ResNet50  (no torch.compile)")
print("=" * 60)

baseline_model = create_model(num_classes=NUM_CLASSES).to(device)
# Fine-tuning: higher LR for the new classification head, lower for the pretrained backbone
baseline_optimizer = torch.optim.Adam(
    [
        {"params": baseline_model.fc.parameters(), "lr": 1e-3},
        {
            "params": [
                p
                for n, p in baseline_model.named_parameters()
                if not n.startswith("fc")
            ],
            "lr": 1e-4,
        },
    ]
)

t0 = time.perf_counter()
baseline_results = train(
    model=baseline_model,
    train_data_loader=train_loader,
    test_data_loader=test_loader,
    optimizer=baseline_optimizer,
    device=device,
    accuracy_fn=accuracy_fn,
    loss_fn=nn.CrossEntropyLoss(),
    epochs=EPOCHS,
)
baseline_time = time.perf_counter() - t0
print(f"\nBaseline total time: {baseline_time:.2f}s")

# ---------------------------------------------------------------------------
# 2. Compiled ResNet50 (torch.compile)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2 / 2  —  Compiled ResNet50  (torch.compile)")
print("=" * 60)

# Create optimizer on the raw model BEFORE compiling so fc is directly accessible
_raw_model = create_model(num_classes=NUM_CLASSES).to(device)
compiled_optimizer = torch.optim.Adam(
    [
        {"params": _raw_model.fc.parameters(), "lr": 1e-3},
        {
            "params": [
                p for n, p in _raw_model.named_parameters() if not n.startswith("fc")
            ],
            "lr": 1e-4,
        },
    ]
)
compiled_model, compile_setup_s, _ = create_compile_model(_raw_model)

t0 = time.perf_counter()
compiled_results = train(
    model=compiled_model,
    train_data_loader=train_loader,
    test_data_loader=test_loader,
    optimizer=compiled_optimizer,
    device=device,
    accuracy_fn=accuracy_fn,
    loss_fn=nn.CrossEntropyLoss(),
    epochs=EPOCHS,
)
compiled_time = time.perf_counter() - t0
print(
    f"\nCompiled total time: {compiled_time:.2f}s  "
    f"(includes {compile_setup_s:.4f}s torch.compile() setup)"
)

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
speedup = baseline_time / compiled_time

print("\n" + "=" * 60)
print(f"{'Comparison':^60}")
print("=" * 60)
print(f"{'Metric':<35} {'Baseline':>10} {'Compiled':>10}")
print("-" * 60)
print(
    f"{'Total train+test time (s)':<35} {baseline_time:>10.2f} {compiled_time:>10.2f}"
)
print(f"{'Speedup (baseline / compiled)':<35} {'—':>10} {speedup:>10.2f}x")
print(
    f"{'Final train accuracy (%)':<35} {baseline_results['train_accuracy'][-1]:>10.2f} {compiled_results['train_accuracy'][-1]:>10.2f}"
)
print(
    f"{'Final test accuracy  (%)':<35} {baseline_results['test_accuracy'][-1]:>10.2f} {compiled_results['test_accuracy'][-1]:>10.2f}"
)
print("=" * 60)

if speedup > 1:
    print(f"\ntorch.compile is {speedup:.2f}x FASTER over {EPOCHS} epochs.")
else:
    print(
        f"\ntorch.compile is {1 / speedup:.2f}x SLOWER over {EPOCHS} epochs "
        f"(compile overhead dominates at this epoch count)."
    )
