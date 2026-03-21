import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable

from timeit import default_timer
import time
from tqdm.auto import tqdm
from pathlib import Path

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import torchmetrics

import matplotlib.pyplot as plt

from common.helper_fucntion import accuracy_fn, print_train_time
from common.device import get_best_device, print_device_info
from lessons.section5_pytorch_computer_vision.common.common import evaluate_model

"""
0. Computer vision libraries in PyTorch
- torchvision - base domain for computer vision in PyTorch
- torchvision.datasets - get datasets and data loading for computer vision
- torchvision.models - get pretrained computer vision models that you can leverage for your own problems
- torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
- torch.utils.data.Dataset - Base dataset class for PyTorch
- torch.utils,data.Dataload - creates a Python iterable over a dataset
"""

"""
1. Getting a dataset
Dataset will be used is FashionMNIST from torchvision.datasets
"""
# Set up training data
train_data = datasets.FashionMNIST(
    root="lessons/section5_pytorch_computer_vision/data",  # where to download data to
    train=True,  # do we want the training dataset?
    download=False,  # do we want to download
    transform=ToTensor(),  # how do we want to transform the data
    target_transform=None,  # how do we want to transform the labels/targets
)

test_data = datasets.FashionMNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None,
)

# Check sample size
print(f"Train data length {len(train_data)} | test data {len(test_data)}")

# See the first training example
image, label = train_data[0]
image: torch.Tensor
class_name = train_data.classes
"""
2. Prepare DataLoader

Currently, FashionMNIST is in the form of PyTorch Datasets
DataLoader turns datasets to a Python iterable
Specifically, we want to turn our data into batches or mini-batches
Why batches? 
1. It is more computationally efficiently, as in , your computing hardware may not be able to load 60000 images in one hit.
So we break it down to 32 images a time (batch size of 32 -- can be changed)
2. It gives our nerual network more chances to update its gradients per epoch
"""
# Set up the batch size hyperparameter
# Larger batch size keeps the GPU fed and reduces per-batch kernel launch overhead
BATCH_SIZE = 32

# Turn data set into iterables (batches)
# pin_memory=True: pre-allocates data in page-locked memory for faster CPU->GPU transfer
# num_workers=4: loads batches in parallel background processes so GPU never waits for data
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

# No need to shuffle test data, it will not be used in training
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

# Check out the train and test dataloader
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(
    f"Length of train_dataloader: {len(train_dataloader)} | test_dataloader {len(test_dataloader)}"
)
print(
    f"Batch size: train_dataloader {train_dataloader.batch_size} | test_dataloader {test_dataloader.batch_size}"
)

# Checkout what is inside train_dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch: torch.Tensor
train_labels_batch: torch.Tensor

print(
    f"one batch of train_features {train_features_batch.shape} | train_labels {train_labels_batch.shape}"
)
# one batch of train_features torch.Size([32, 1, 28, 28]) | train_labels torch.Size([32])

"""
5. Set up device agnostic-code (if GPU is avaliable)
"""
device = get_best_device()
print_device_info(device=device)

"""
7. Model 2: Building a Convolutional Neural Network (CNN)

CNN also called ConvNets
CNN are know for their capabilities to find patterns in visual data

Resource: https://poloclub.github.io/cnn-explainer/
"""


# Create a Convolutional Neural Network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture that replicate the Tiny VGG (Visual Geometry Group)
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        # --- Conv Block 1 ---
        # Input tensor shape: [batch, input_shape, 28, 28]  (e.g. [32, 1, 28, 28] for FashionMNIST)
        #
        # Conv2d spatial formula: H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
        #
        # Conv2d #1:  H_out = floor((28 + 2*1 - 3) / 1 + 1) = 28  →  [batch, hidden_units, 28, 28]
        # ReLU: no shape change, applies element-wise non-linearity
        # Conv2d #2:  H_out = 28 (same padding=1, kernel=3, stride=1) →  [batch, hidden_units, 28, 28]
        # ReLU: no shape change
        # MaxPool2d (kernel=2, stride defaults to kernel=2):
        #             H_out = floor(28 / 2) = 14   →  [batch, hidden_units, 14, 14]
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,  # 1 for grayscale (FashionMNIST), 3 for RGB
                out_channels=hidden_units,  # number of learned filters / feature maps
                kernel_size=3,  # 3×3 sliding window
                stride=1,  # move window 1 pixel at a time
                padding=1,  # pad borders so spatial size is preserved after conv
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,  # halves spatial dimensions: 28×28 → 14×14
            ),
        )

        # --- Conv Block 2 ---
        # Input: [batch, hidden_units, 14, 14]
        # Conv2d #1 & #2 (padding=1, kernel=3, stride=1): shape unchanged → [batch, hidden_units, 14, 14]
        # MaxPool2d (kernel=2): 14×14 → 7×7  →  [batch, hidden_units, 7, 7]
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14×14 → 7×7
        )

        # --- Classifier (fully-connected head) ---
        # Flatten: [batch, hidden_units, 7, 7] → [batch, hidden_units * 7 * 7]
        #   in_features = hidden_units * 7 * 7
        #     - hidden_units: number of feature maps coming out of the last conv block
        #     - 7 * 7:        spatial size after two MaxPool2d(kernel=2) on a 28×28 input
        #                     (28 → 14 → 7)
        # Linear: maps the flattened vector to class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 7 * 7,  # must match flattened conv output
                out_features=output_shape,  # one logit per class
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28]
        x = self.conv_block_1(x)  # → [batch, hidden_units, 14, 14]
        x = self.conv_block_2(x)  # → [batch, hidden_units,  7,  7]
        x = self.classifier(x)  # → [batch, output_shape]
        return x


torch.manual_seed(42)

model_2 = FashionMNISTModelV2(
    input_shape=1,  # number of color channels in the image, FashionMNIST has only 1 channel
    hidden_units=10,
    output_shape=len(class_name),
).to(device)


"""
7.1 Stepping through nn.Conv2d()
"""
torch.manual_seed(42)

# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64)) # batch, color channel, hight, width
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
# print(f"Test image: \n{test_image}")

# Create a Conv2d layer
conv_layer = nn.Conv2d(
    in_channels=3,
    out_channels=10,
    kernel_size=3,
    stride=1,
    padding=0,
)

# Pass the data throught the convolutional layer
conv_output = conv_layer(test_image)
print(f"after the conv_layer, conv_output shape is {conv_output.shape}")

"""
7.2 Stepping throught nn.MaxPool2d()
"""
# Use the created "simulation" images to pass through the MaxPool2d
print(f"\nImage batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
print(f"Single image shape with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

# Create nn.MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv layer
test_image_through_conv = conv_layer(test_image)
print(f"test image through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

# Create a smaller tensor so easy to visualize, with similar tensor
torch.manual_seed(42)
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"\nrandom_tensor: \n{random_tensor}")
print(f"random_tensor shape: {random_tensor.shape}")

# Create a nother MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass the random tensor throught the max pool layer
max_pool_tensor: torch.Tensor = max_pool_layer(random_tensor)

print(f"\n max_pool_tensor: \n {max_pool_tensor}")
print(f"max_pool_tensor shape: {max_pool_tensor.shape}")

