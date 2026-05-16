import torch
from torchvision.transforms import v2
from pathlib import Path


from torchinfo import summary

from going_modular.pytorch_project import data_setup
from common.device import get_best_device, print_device_info

from lessons.section9_pytorch_experiment_tracking.common.common_functions import (
    create_efficientnet_b0_model,
    create_efficientnet_b2_model,
)

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

# Example of creating a SummaryWriter instance with the create_summary_writer() function
# example_writer = create_summary_writer(
#     experiment_name="data_10_percent", model_name="efficientnet_b0", extra="5_epochs"
# )
# print(f"Example log directory: {example_writer.log_dir}")

"""
7. Setting up a series of modeling experiments
"""

"""
7.1 What experiments to run? (different hyperparameters)
- Number of epochs
- Number of hidden units
- amount of data to train on
- Learning rate
- different kinds of augmentation
- Choose a different model architecture

This is why transfer learning is so useful, it allows us to quickly iterate through different experiments by only changing a few lines of code.
"""

"""
7.2 What experiments to run in this code? (make it simple currently)
1. Model size - efficientnet_b0 vs efficientnet_b2 (number of parameters - 5.3 million vs 9.2 million)
2. Dataset size - 10% vs 20% (pizza, steak, sushi)
3. Training epochs - 5 vs 10 epochs
"""

"""
7.3 Download the data (already done in part 1, so we can skip this step)
10% dir - going_modular/data/pizza_steak_sushi
20% dir - lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased
"""

"""
7.4 Transform the data and create DataLoaders

1. Resize (224, 224) - required for efficientnet_b0 and efficientnet_b2
2. Convert to tensor, value between [0, 1]
3. Normalize (efficientnet_b0 and efficientnet_b2 were trained on imagenet, so we will use the imagenet mean and std for normalization)
"""
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

print(f"Training directory for 10% data: {train_dir_10_percent}")
print(f"Training directory for 20% data: {train_dir_20_percent}")
print(f"Testing directory: {test_dir}")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32

# Set up transforms (same for both)
manual_transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_dataloader_10_percent, test_dataloader, class_names = (
    data_setup.create_dataloaders(
        train_dir=str(train_dir_10_percent),
        test_dir=str(test_dir),
        train_transform=manual_transform,
        test_transform=manual_transform,
        batch_size=BATCH_SIZE,
    )
)

train_dataloader_20_percent, _, _ = data_setup.create_dataloaders(
    train_dir=str(train_dir_20_percent),
    test_dir=str(test_dir),
    train_transform=manual_transform,
    test_transform=manual_transform,
    batch_size=BATCH_SIZE,
)

print(
    f"Number of batches in 10% training dataloader: {len(train_dataloader_10_percent)}"
)
print(
    f"Number of batches in 20% training dataloader: {len(train_dataloader_20_percent)}"
)
print(f"Number of batches in test dataloader: {len(test_dataloader)}")
print(f"Class names: {class_names}")

# Use create_efficientnet_b0_model() and create_efficientnet_b2_model() to create models
efficientnet_b0_model = create_efficientnet_b0_model(num_classes=len(class_names)).to(
    device
)
efficientnet_b2_model = create_efficientnet_b2_model(num_classes=len(class_names)).to(
    device
)

summary(
    model=efficientnet_b0_model,
    input_size=(1, 3, 224, 224),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    col_width=20,
    row_settings=["var_names"],
)

summary(
    model=efficientnet_b2_model,
    input_size=(1, 3, 224, 224),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    col_width=20,
    row_settings=["var_names"],
)
