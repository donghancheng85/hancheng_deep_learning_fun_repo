import torch
from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time

from lessons.section11_pytorch_model_deployment.model_training.vit_b_16_model_creater import (
    create_vit_b16_model,
)

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# ── Data ──────────────────────────────────────────────────────────────────────────────
data_path = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)
test_dir = data_path / "test"

# Get all test data paths
print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
print(f"[INFO] Found {len(test_data_paths)} test data paths.")
print(f"[INFO] First 5 test data paths: {test_data_paths[:5]}")
