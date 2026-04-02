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
import os

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

Transforms help us transform our data into a format which is ready for PyTorch to process and data augmentation
"""
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

transformed = data_transform(img)
assert isinstance(transformed, torch.Tensor)
print(
    f"Transformed image shape: {transformed.shape}"
)  # -> torch.Size([3, 64, 64]) = [C, H, W]


def plot_transformed_image(image_paths, transform, n=3, seed=None):
    """
    Plot n random images in a single figure with 2 columns:
      - Left column: original image
      - Right column: transformed image
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, n * 3))

    for row, image_path in enumerate(random_image_paths):
        with Image.open(image_path) as f:
            class_name = image_path.parent.stem

            # Left: original image
            axes[row, 0].imshow(f)
            axes[row, 0].set_title(f"Original | {class_name}\nSize: {f.size}")
            axes[row, 0].axis(False)

            # Right: transformed image
            # transform returns [C, H, W] tensor; permute to [H, W, C] for matplotlib
            transformed_image = transform(f)
            assert isinstance(transformed_image, torch.Tensor)
            axes[row, 1].imshow(transformed_image.permute(1, 2, 0))
            axes[row, 1].set_title(
                f"Transformed | {class_name}\nShape: {transformed_image.shape}"
            )
            axes[row, 1].axis(False)


plot_transformed_image(
    image_paths=image_path_list, transform=data_transform, n=3, seed=42
)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/a_line_190_transformed_images.png"
)

"""
4. Option 1: Loading image data using ImageFolder, it is a pre-built dataset for 
   loading image data with a standard structure (class folders with images inside)

   We can load image classification data with torchvision.datasets.ImageFolder
"""

# Use ImageFolder to load the data
train_data = datasets.ImageFolder(
    root=train_dir,
    transform=data_transform,  # apply the transform we created to each image when loading
    target_transform=None,  # we can also apply a transform to the target labels if we want, but we'll leave it as is for now
)
test_data = datasets.ImageFolder(
    root=test_dir,
    transform=data_transform,  # apply the transform we created to each image when loading
)
print(f"\nNumber of training samples: {len(train_data)}")

# Get the class names from the training data
class_name = train_data.classes
print(f"Class names: {class_name}")

# Get class name as dict
class_dict = train_data.class_to_idx
print(f"Class name to index mapping: {class_dict}")
# Class name to index mapping: {'pizza': 0, 'steak': 1, 'sushi': 2}

# Check the lenght of datasets
print(f"Number of training samples: {len(train_data)}")
print(f"Number of testing samples: {len(test_data)}")

# index on the train_data to get a sample
image, label = train_data[0]
print(
    f"Image shape: {image.shape} | Label: {label} | Class name: {class_name[label]} | data type: {image.dtype} | label data type: {type(label)}"
)

# Using matplotlib to visualize the image and its label, first rearrange the image from [C, H, W] to [H, W, C] for matplotlib
image_permute = image.permute(1, 2, 0)
print(
    f"Original image shape: {image.shape} -> [C, H, W] | Permuted image shape: {image_permute.shape} -> [H, W, C]"
)
plt.figure(figsize=(6, 6))
plt.imshow(image_permute)
plt.axis(False)
plt.title(f"Class: {class_name[label]} | Shape: {image.shape} -> [C, H, W]")
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/a_line_249_visualize_image_from_imagefolder.png"
)

"""
4.1 Turn ImageFolder dataset into DataLoader for training
"""
# Trun ImageFolder dataset into DataLoader
BATCH_SIZE = 1
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # shuffle the data for training
    num_workers=1,  # use all available CPU cores for data loading
    pin_memory=True,  # pin memory for faster data transfer to GPU
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    num_workers=1,
    shuffle=False,  # no need to shuffle test data, it will not be used in training
)

# Check the dataloader
print(f"Number of batches in train dataloader: {len(train_dataloader)}")
print(f"Number of batches in test dataloader: {len(test_dataloader)}")
image_batch, label_batch = next(iter(train_dataloader))

# Batch size of 1
print(
    f"Image batch shape: {image_batch.shape} -> [batch_size, channels, height, width] | Label batch shape: {label_batch.shape} -> [batch_size]"
)  # Image batch shape: torch.Size([1, 3, 64, 64]) | Label batch shape: torch.Size([1]  )
