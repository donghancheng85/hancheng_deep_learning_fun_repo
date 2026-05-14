import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torchvision import transforms
from datetime import datetime
import os

from torchinfo import summary

from going_modular.pytorch_project import data_setup, engine, download_data, utils
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, create_summary_writer, pred_and_plot_image

from lessons.section9_pytorch_experiment_tracking.common.common_functions import (
    create_efficientnet_b0_model,
    create_efficientnet_b2_model,
)

# Set up device
device = get_best_device()
print_device_info(device)

# 1. Re-create the model architecture with the same custom classifier head
# (3 output classes: pizza, steak, sushi)
# Use weights=None to skip loading pretrained ImageNet weights — they will be
# fully overwritten by load_state_dict anyway, so loading them is wasteful.
model = torchvision.models.efficientnet_b2(weights=None)
in_features: int = model.classifier[-1].in_features  # type: ignore[union-attr]
model.classifier[-1] = nn.Linear(in_features=in_features, out_features=3)
model = model.to(device)

# 2. Load the saved state_dict into the model
model_path = Path("lessons/section9_pytorch_experiment_tracking/models/09_effnetb2_data_20_percent_10_epochs.pth")
model.load_state_dict(torch.load(f=model_path, map_location=device))
print(f"[INFO] Loaded model from: {model_path}")

# 3. Put the model in eval mode for inference
model.eval()
print(f"[INFO] Model is ready for inference: {model.__class__.__name__}")

