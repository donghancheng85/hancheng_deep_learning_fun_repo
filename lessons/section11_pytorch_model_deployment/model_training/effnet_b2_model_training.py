import torch
import torchvision
from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time

from going_modular.pytorch_project.train import BATCH_SIZE
from lessons.section11_pytorch_model_deployment.model_training.effnet_b2_model_creater import (
    create_effnet_b2_model,
)

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Data ──────────────────────────────────────────────────────────────────────────────
data_path = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)
train_dir = data_path / "train"
test_dir = data_path / "test"

# ── Model ─────────────────────────────────────────────────────────────────────────────
model, pretrained_transform = create_effnet_b2_model(num_classes=3, seed=42)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=pretrained_transform,
    test_transform=pretrained_transform,
    batch_size=BATCH_SIZE,
)

print(f"Class names: {class_names}")
print(f"Train batches: {len(train_dataloader)} | Test batches: {len(test_dataloader)}")
