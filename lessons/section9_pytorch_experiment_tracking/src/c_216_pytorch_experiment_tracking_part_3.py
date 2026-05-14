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
from common.helper_fucntion import accuracy_fn, set_seeds, create_summary_writer

from lessons.section9_pytorch_experiment_tracking.common.common_functions import (
    create_efficientnet_b0_model,
    create_efficientnet_b2_model,
)

# set up device
device = get_best_device()
print_device_info(device)

# Set up paths
data_path_10_percent = Path("going_modular/data/pizza_steak_sushi")
data_path_20_percent = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)

# Set up training dir
train_dir_10_percent = data_path_10_percent / "train"
train_dir_20_percent = data_path_20_percent / "train"

# Set up test dir (same for both)
test_dir = data_path_10_percent / "test"

BATCH_SIZE = 32

# Use each model's recommended pretrained transforms (correct native input resolution)
# EfficientNet-B0 was pretrained at 224x224; EfficientNet-B2 at 260x260 (crops to 288)
effnetb0_transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
effnetb2_transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()

# Create dataloaders for effnetb0
train_dataloader_10_percent_b0, test_dataloader_b0, class_names = (
    data_setup.create_dataloaders(
        train_dir=str(train_dir_10_percent),
        test_dir=str(test_dir),
        train_transform=effnetb0_transform,
        test_transform=effnetb0_transform,
        batch_size=BATCH_SIZE,
    )
)
train_dataloader_20_percent_b0, _, _ = data_setup.create_dataloaders(
    train_dir=str(train_dir_20_percent),
    test_dir=str(test_dir),
    train_transform=effnetb0_transform,
    test_transform=effnetb0_transform,
    batch_size=BATCH_SIZE,
)

# Create dataloaders for effnetb2
train_dataloader_10_percent_b2, test_dataloader_b2, _ = data_setup.create_dataloaders(
    train_dir=str(train_dir_10_percent),
    test_dir=str(test_dir),
    train_transform=effnetb2_transform,
    test_transform=effnetb2_transform,
    batch_size=BATCH_SIZE,
)
train_dataloader_20_percent_b2, _, _ = data_setup.create_dataloaders(
    train_dir=str(train_dir_20_percent),
    test_dir=str(test_dir),
    train_transform=effnetb2_transform,
    test_transform=effnetb2_transform,
    batch_size=BATCH_SIZE,
)

"""
7.6 Create experiments and set up training code
"""
# Create epoch list
num_epochs = [5, 10]

# Create model list
models = ["effnetb0", "effnetb2"]

# Dataloaders grouped by model so each model uses its correct native transform
train_dataloaders = {
    "effnetb0": {
        "data_10_percent": train_dataloader_10_percent_b0,
        "data_20_percent": train_dataloader_20_percent_b0,
    },
    "effnetb2": {
        "data_10_percent": train_dataloader_10_percent_b2,
        "data_20_percent": train_dataloader_20_percent_b2,
    },
}
test_dataloaders = {
    "effnetb0": test_dataloader_b0,
    "effnetb2": test_dataloader_b2,
}

# Keep track of experiment numbers
experiment_number = 0

# Loop each data loader name
for train_dataloader_name in ["data_10_percent", "data_20_percent"]:
    # Loop through the epochs
    for epochs in num_epochs:
        # Loop through the models
        for model_name in models:
            experiment_number += 1
            print(
                f"[INFO] Experiment number: {experiment_number} \n"
                f"[INFO] Model: {model_name} \n"
                f"[INFO] Epochs: {epochs} \n"
                f"[INFO] Dataloader: {train_dataloader_name}"
            )

            # Set the seed for each experiment to ensure reproducibility and fair comparison
            set_seeds(seed=42)

            # Select the correct dataloaders for this model (each uses its native pretrained transform)
            train_dataloader = train_dataloaders[model_name][train_dataloader_name]
            test_dataloader = test_dataloaders[model_name]

            # Create a model
            if model_name == "effnetb0":
                model = create_efficientnet_b0_model(num_classes=len(class_names)).to(
                    device
                )
            elif model_name == "effnetb2":
                model = create_efficientnet_b2_model(num_classes=len(class_names)).to(
                    device
                )

            # Create a loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Create a summary writer
            experiment_name = f"{model_name}_{train_dataloader_name}_{epochs}_epochs"

            writer = create_summary_writer(
                experiment_name=train_dataloader_name,
                model_name=model_name,
                extra=f"{epochs}_epochs",
            )

            # Train the model and save results to TensorBoard
            engine.train_for_summarywriter(
                model=model,
                train_data_loader=train_dataloader,
                test_data_loader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                accuracy_fn=accuracy_fn,
                writer=writer,
            )

            # Save the model
            utils.save_model(
                model=model,
                save_path=Path("lessons/section9_pytorch_experiment_tracking/models"),
                model_name=f"09_{model_name}_{train_dataloader_name}_{epochs}_epochs.pth",
            )

            print("-" * 50)

"""
8. Visualize training results in TensorBoard
Run the following command in your terminal to launch TensorBoard and visualize the results of your experiments:
tensorboard --logdir lessons/section9_pytorch_experiment_tracking/runs

Let use the effnetb2 model as the best one
"""
