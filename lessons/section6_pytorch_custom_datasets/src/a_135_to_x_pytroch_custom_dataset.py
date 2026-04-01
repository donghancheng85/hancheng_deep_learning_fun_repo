import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.device import get_best_device, print_device_info

import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

"""
How we want to get our own data to PyTorch

One of the ways is through: custom datasets
"""

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
zip_path = data_path / "pizza_steak_sushi.zip"

# Download zip if it doesn't exist yet
if zip_path.exists():
    print(f"{zip_path} already exists... skipping download")
else:
    data_path.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        response = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip"
        )
        print("Downloading data...")
        f.write(response.content)

# Unzip if the extracted folder doesn't exist yet
if image_path.is_dir():
    print(f"{image_path} already exists... skipping unzip")
else:
    image_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping data...")
        zip_ref.extractall(image_path)

"""
2. Becoming one with data (data preparation and exploration)
"""


def walk_through_dir(dir_path: Path):
    """
    Walk through dir_path returning its contents
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}' "
        )


walk_through_dir(image_path)

# Setup training and testing path
train_dir = image_path / "train"
test_dir = image_path / "test"

"""
2.1 visualizing image
1. Get all the image paths
2. Pick a random image path using Python's random.choice()
3. Get the image class name (dir name) pathlib.Path.parent.stem
4. open the image with Python's pillow lib
5. show the imange and print metadata
"""
random.seed(42)

# 1. Get all image path
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Pick a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# 3. Get the image class from the path name (name of the dir)
image_class_name = random_image_path.parent.stem
print(f"Image class name: {image_class_name}")

# 4. Open the image with Python's pillow lib
img = Image.open(random_image_path)

# 5. Print metdata
print(f"Random image path: {random_image_path}")
print(f"Image class name: {image_class_name}")
print(f"image height: {img.height}, image width: {img.width}, image mode: {img.mode}")

# 6. Visualize the image with matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(
    np.array(img)
)  # need to convert PIL image to numpy array for matplotlib to visualize
plt.title(
    f"Class: {image_class_name} | Shape: {np.array(img).shape} -> (height, width, color_channels)"
)
plt.axis(False)
plt.tight_layout()
plt.savefig("lessons/section6_pytorch_custom_datasets/src/a_line_106_random_image.png")

"""
3, Transforming data and making it ready for PyTorch

Steps:
1. Turn the image into tensors
2. Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader
"""

"""
3.1 Tranforming data with torchvision.transforms
"""
# Write a transform for image
data_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # resize all images to 64x64
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # data augmentation, randomly flip some images horizontally
        transforms.ToTensor(),  # convert image to tensor, range [0, 255] -> [0.0, 1.0]
    ]
)

transformed = data_transform(img)
assert isinstance(transformed, torch.Tensor)
print(
    f"Transformed image shape: {transformed.shape}"
)  # -> torch.Size([3, 64, 64]) = [C, H, W]
