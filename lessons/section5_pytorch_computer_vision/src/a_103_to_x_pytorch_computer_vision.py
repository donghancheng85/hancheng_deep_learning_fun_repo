import torch
from torch import nn
from torch.utils.data import DataLoader

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
print(f"Class names {train_data.classes}")
print(f"Class to index {train_data.class_to_idx}")
print(
    f"Type of image {type(image)} | shape {image.shape} -> [C, H, W] | label class {class_name[label]}"
)

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
for i in range(1, rows * cols + 1):
    random_index = torch.randint(low=0, high=len(train_data), size=(1,)).item()
    img, label = train_data[int(random_index)]
    img: torch.Tensor
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(dim=0), cmap="gray")
    plt.title(class_name[label])
    plt.axis(False)
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/a_line_72_visualize_random_train_data.png"
)

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
BATCH_SIZE = 32

# Turn data set into iterables (batches)
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# No need to shuffle test data, it will not be used in training
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Check out the train and test dataloader
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(
    f"Length of train_dataloader: {len(train_dataloader)} | test_dataloader {len(test_dataloader)}"
)
print(
    f"Batch size: train_dataloader {train_dataloader.batch_size} | test_dataloader {test_dataloader.batch_size}"
)

# Checkout what is inside train_dateloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch: torch.Tensor
train_labels_batch: torch.Tensor

print(
    f"one batch of train_features {train_features_batch.shape} | train_labels {train_labels_batch.shape}"
)
# one batch of train_features torch.Size([32, 1, 28, 28]) | train_labels torch.Size([32])

# Visual a sample (how to interact with dataloader)
torch.manual_seed(42)
random_index = torch.randint(low=0, high=len(train_features_batch), size=(1,)).item()
img_batch, label_batch = (
    train_features_batch[int(random_index)],
    train_labels_batch[int(random_index)],
)
plt.figure()
plt.imshow(img_batch.squeeze(dim=0), cmap="gray")
plt.title(class_name[label_batch])
plt.axis(False)
print(f"Image size: {img_batch.shape}")
print(f"Label: {label}, label size {label_batch.shape}")
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/a_line_128_visualize_img_in_train_dataloader.png"
)
