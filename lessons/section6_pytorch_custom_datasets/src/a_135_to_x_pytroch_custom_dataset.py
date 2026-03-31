import torch
from torch import nn

from common.device import get_best_device, print_device_info

import requests
import zipfile
from pathlib import Path

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
