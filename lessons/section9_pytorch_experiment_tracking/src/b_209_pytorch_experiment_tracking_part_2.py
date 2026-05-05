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

from going_modular.pytorch_project import data_setup, engine, download_data
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds

# set up device
device = get_best_device()
print_device_info(device)

"""
6. Create a function to prepare a SummaryWriter() instance

We want to different models to different folders

One experiment = one folder

We want to track:
 - date/timestamp
 - Experiment name
 - model name
 - hyperparameters
 - training/validation metrics
 - any other relevant information

name will be like:
runs/YYYY-MM-DD-HH-MM-SS/experiment-name/model-name/hyperparameters(extras)
"""


def create_summary_writer(
    experiment_name: str, model_name: str, extra: str | None = None
) -> SummaryWriter:
    """Creates a SummaryWriter instance with a specific folder structure for organizing experiments."""
    # Get timestamp of current date in reverse order (YYYY-MM-DD-HH-MM-SS)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if extra:
        log_dir = os.path.join(
            "lessons/section9_pytorch_experiment_tracking/runs",
            timestamp,
            experiment_name,
            model_name,
            extra,
        )
    else:
        log_dir = os.path.join(
            "lessons/section9_pytorch_experiment_tracking/runs",
            timestamp,
            experiment_name,
            model_name,
        )
    print(f"[INFO] Creating SummaryWriter with log directory: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

example_writer = create_summary_writer(
    experiment_name="data_10_percent", model_name="efficientnet_b0", extra="5_epochs"
)
print(f"Example log directory: {example_writer.log_dir}")
