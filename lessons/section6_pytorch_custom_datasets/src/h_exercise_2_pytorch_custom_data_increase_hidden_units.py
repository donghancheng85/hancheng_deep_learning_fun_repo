import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGG,
)

from timeit import default_timer
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Recreate the data loading functions we built in sections 1, 2, 3 and 4. You should have train and test DataLoader's ready to use.
"""

# get best device for training
device = get_best_device()
print_device_info(device)

# Set up path to a data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

# Create train transforms
train_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

test_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

# Use ImageFolder to create datasets
train_dataset = datasets.ImageFolder(
    root=train_dir, transform=train_transform, target_transform=None
)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Create DataLoaders
BATCH_SIZE = 32
# Cap workers to avoid exhausting /dev/shm on Linux, which causes a segfault.
# os.cpu_count() can be large; 2-4 workers is typically safe for image datasets.
NUM_WORKERS = min(os.cpu_count() or 1, 4)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

"""
6. Double the number of hidden units in your model and train it for 20 epochs, what happens to the results?
"""

model_0 = TinyVGG(
    in_features=3,
    hidden_units=20,
    out_features=len(train_dataset.classes),
)
