import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGG,
)

from typing import Tuple, Dict, List
from pathlib import Path
import pathlib
import os
import random
from PIL import Image
from timeit import default_timer
import matplotlib.pyplot as plt

"""
0. Setting up device-agnostic code
"""
device = get_best_device()
print_device_info(device)

"""
1. Setup data paths
   (assumes pizza_steak_sushi data already downloaded and unzipped by a_135_to_x_*.py)
"""
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

"""
9. TinyVGG model with data augmentation
"""
"""
9.1 Create data augmentation transforms using torchvision.transforms
"""
train_transform_trivial = v2.Compose(
    [
        v2.Resize((64, 64)),
        v2.TrivialAugmentWide(num_magnitude_bins=31),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
test_transform_simple = v2.Compose(
    [v2.Resize((64, 64)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
)

"""
9.2 Create datasets and dataloaders with the data augmentation transforms
"""
train_dataset_augmented = datasets.ImageFolder(
    root=train_dir, transform=train_transform_trivial
)
test_dataset_simple = datasets.ImageFolder(
    root=test_dir, transform=test_transform_simple
)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

# Just for reproducibility, set random seeds
torch.manual_seed(42)

train_dataloader = DataLoader(
    dataset=train_dataset_augmented,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
test_dataloader = DataLoader(
    dataset=test_dataset_simple,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


"""
9.3 Create TinyVGG model, loss function and optimizer"""
torch.manual_seed(42)
model_1 = TinyVGG(
    in_features=3,
    hidden_units=10,
    out_features=len(train_dataset_augmented.classes),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)

"""
9.4 Train the model with data augmentation and plot the loss curves
"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)
NUM_EPOCHS = 10
start_time = default_timer()
model_1_results = train(
    model=model_1,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    accuracy_fn=accuracy_fn,
    epochs=NUM_EPOCHS,
)
end_time = default_timer()
print_train_time(start_time, end_time, device=device)

print(model_1_results)
plot_loss_curves(model_1_results)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/d_line_122_model_1_loss_curves.png"
)

"""
Sample output:
{'model_name': 'TinyVGG', 'train_loss': [1.1037115156650543, 1.0763073936104774, 1.0745355859398842, 1.117841899394989, 1.081290602684021, 1.0965303480625153, 1.0544301122426987, 1.0637394040822983, 1.0622591599822044, 1.0040317848324776, 0.9607392475008965, 0.9803609922528267, 0.9505485594272614, 0.9474351331591606, 0.9690867587924004, 0.8865455463528633, 0.9473721086978912, 0.9878225922584534, 0.9086165204644203, 0.922882191836834], 'train_accuracy': [25.0, 42.578125, 42.578125, 30.46875, 47.265625, 38.671875, 52.34375, 51.5625, 44.140625, 43.75, 52.34375, 54.296875, 56.640625, 55.078125, 57.421875, 61.328125, 57.03125, 46.484375, 58.984375, 59.765625], 'test_loss': [1.1013043721516926, 1.1317500869433086, 1.1680513223012288, 1.1571898857752483, 1.1535391012827556, 1.1407412091890972, 1.1264049609502156, 1.122567613919576, 1.0951989491780598, 0.9865210254987081, 1.0291932026545207, 1.0558059215545654, 1.0353090365727742, 1.0455910762151082, 1.021833101908366, 1.0866369406382244, 1.0543920199076335, 1.0294636487960815, 1.088698108990987, 1.0232189893722534], 'test_accuracy': [26.041666666666668, 26.041666666666668, 26.041666666666668, 26.041666666666668, 26.041666666666668, 29.166666666666668, 32.291666666666664, 31.25, 34.375, 44.31818181818181, 41.47727272727273, 32.10227272727273, 42.42424242424242, 39.3939393939394, 35.13257575757576, 39.3939393939394, 35.22727272727273, 39.29924242424242, 34.18560606060606, 44.41287878787879]}
"""

"""
10. Compare model results with and without augmentation

Some approaches:
1. Hard code
2. Pytorch + TensorBoard
3. Weights + Biases (wandb)
4. MLflow
"""

"""
10.1 Load model_0 results (no augmentation) saved by c_152_*.py
"""
_results_path = Path(
    "lessons/section6_pytorch_custom_datasets/src/c_model_0_results.json"
)
with open(_results_path) as f:
    model_0_results = json.load(f)

print("model_0 results loaded:", model_0_results)

"""
10.2 Plot model_0 vs model_1 loss and accuracy curves side by side for comparison
    model_0 = TinyVGG without augmentation (c_152_*.py)
    model_1 = TinyVGG with TrivialAugmentWide (this file)

Note: the two models were trained for different numbers of epochs (10 vs 20),
so we align by epoch index — the x-axis simply counts epochs from 1.
"""
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# --- Loss ---
ax = axes[0]
for results, label in [
    (model_0_results, "model_0 (no augment)"),
    (model_1_results, "model_1 (augmented)"),
]:
    epochs_range = range(1, len(results["train_loss"]) + 1)
    ax.plot(epochs_range, results["train_loss"], linestyle="--", label=f"{label} train")
    ax.plot(epochs_range, results["test_loss"], linestyle="-", label=f"{label} test")
ax.set_title("Loss Comparison: model_0 vs model_1")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.grid()

# --- Accuracy ---
ax = axes[1]
for results, label in [
    (model_0_results, "model_0 (no augment)"),
    (model_1_results, "model_1 (augmented)"),
]:
    epochs_range = range(1, len(results["train_accuracy"]) + 1)
    ax.plot(
        epochs_range, results["train_accuracy"], linestyle="--", label=f"{label} train"
    )
    ax.plot(
        epochs_range, results["test_accuracy"], linestyle="-", label=f"{label} test"
    )
ax.set_title("Accuracy Comparison: model_0 vs model_1")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/d_line_199_comparison_model_0_vs_model_1.png"
)

"""
11. Save model_1 weights for use in prediction / comparison in other files
"""
_SAVE_DIR = Path("lessons/section6_pytorch_custom_datasets/src")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_PATH = _SAVE_DIR / "d_model_1.pth"
torch.save(model_1.state_dict(), _MODEL_PATH)
print(f"model_1 saved to: {_MODEL_PATH}")
