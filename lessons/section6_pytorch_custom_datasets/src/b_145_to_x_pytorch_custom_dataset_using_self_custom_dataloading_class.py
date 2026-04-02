import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from common.device import get_best_device, print_device_info

from typing import Tuple, Dict, List
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

"""
0. Setting up device-agnostic code
"""
device = get_best_device()
print_device_info(device)

"""
1. Get data

Dataset will be used here is a subset of Food101 dataset
3 classes and 10% of the images

When starting out ML project, it's important to try things on a small scale then increase
when necessary

Note: we are having standard image class structure
"""
# Set up path to a data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"

# Setup training and testing path
train_dir = image_path / "train"
test_dir = image_path / "test"

# Write a transform for image
data_transform = transforms.Compose(
    [
        transforms.Resize(
            (64, 64)
        ),  # resize all images to 64x64, this is a hyperparameter you can change
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # data augmentation, randomly flip some images horizontally
        transforms.ToTensor(),  # convert image to tensor, range [0, 255] -> [0.0, 1.0]
    ]
)

"""
5. Option 2: Loading Image data with a custom dataset class
This is the most flexible option but also requires the most code to set up and performance 
may not be as good as using a built-in dataset class like ImageFolder

1. Want to be able to load image from file
2. Get class names from dataset
3. Get classes as dictionary from dataset
"""

"""
5.1 Creating a helper to get class names

Steps:
1. Get the class names using os.scandir() to scan the training directory for subdirectories (each subdirectory represents a class).
2. Raise an error if no class names are found (i.e., if there are no subdirectories in the training directory).
3. Turn the class names into a list and dictionary and return them.
"""

# Setup target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted(
    [entry.name for entry in os.scandir(target_directory) if entry.is_dir()]
)
print(f"Class names found: {class_names_found}")

def find_classes(directory: str | Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        directory (str | Path): The root directory of the dataset.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing a list of class names and a dictionary mapping class names to indices.
    """
    # Get the class names from the target directory
    class_names_found = sorted(
        [entry.name for entry in os.scandir(directory) if entry.is_dir()]
    )

    # Raise an error if no class names are found
    if not class_names_found:
        raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")

    # Create a mapping of class names to indices
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names_found)}

    return class_names_found, class_to_idx

class_names, class_to_idx = find_classes(target_directory)
print(f"Class names: {class_names} | Class to index mapping: {class_to_idx}")
