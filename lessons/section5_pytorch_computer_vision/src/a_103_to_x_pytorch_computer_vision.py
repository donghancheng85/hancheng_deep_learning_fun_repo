import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

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
    root="lessons/section5_pytorch_computer_vision/data", # where to download data to
    train=True, # do we want the training dataset?
    download=False, # do we want to download
    transform=ToTensor(), # how do we want to transform the data
    target_transform=None # how do we want to transform the labels/targets
)

test_data = datasets.FashionMNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None
)

# Check sample size
print(f"Train data length {len(train_data)} | test data {len(test_data)}")

# See the first training example
image, label = train_data[0]
image: torch.Tensor
class_name = train_data.classes
print(f"Class names {train_data.classes}")
print(f"Class to index {train_data.class_to_idx}")
print(f"Type of image {type(image)} | shape {image.shape} -> [C, H, W] | label class {class_name[label]}")

"""
1.2 Visualizing the data
"""
plt.imshow(image.squeeze(dim=0), cmap="gray")
plt.title(class_name[label])
plt.axis(False)
plt.savefig("lessons/section5_pytorch_computer_vision/src/a_line_55_visualize_data.png")

# plot more random images, so we have a better idea what the data contains
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_index = torch.randint(low=0, high=len(train_data), size=(1,)).item()
    img, label = train_data[int(random_index)]
    img: torch.Tensor
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(dim=0), cmap="gray")
    plt.title(class_name[label])
    plt.axis(False)
plt.savefig("lessons/section5_pytorch_computer_vision/src/a_line_72_visualize_random_train_data.png")
