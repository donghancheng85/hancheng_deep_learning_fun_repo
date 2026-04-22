from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
from torchinfo import summary

# Import the going_modular directory, download it from GitHub if it doesn't work
from going_modular.pytorch_project import data_setup, engine

from common.device import get_best_device, print_device_info

"""
Pytorch transfer learning is a technique where you take a pre-trained model (trained on a large dataset) and 
fine-tune it on a smaller, task-specific dataset. This allows you to leverage the knowledge learned by the 
pre-trained model, which can lead to better performance and faster training times, especially when you have limited data.
"""

# set up device
device = get_best_device()
print_device_info(device)

"""
8.1 Get pizza, steak, sushi data

Already exist in going_modular/data/pizza_steak_sushi, 
you can use the code in going_modular/get_data.py to download and prepare the data if you don't have it already.
"""
# Setup path to data folder
data_path = Path("going_modular/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"


"""
8.2 Create datasets and DataLoaders

Note the transform, need to fit the pre-trained model's expected input size (e.g. 224x224 for ResNet, 299x299 for Inception).

How to transform the data?
With current version torchvision, two ways to do it:
1. Manually create a transform pipeline using torchvision.transforms.Compose and apply it to the dataset.
2. Automatically create - the transforms are defined by the model you want to use, you can get them from torchvision.models.get_model_weights 
and torchvision.models.get_model_transforms.

Important: when using pre-trained models, it's important that the data (include custom data) is transformed 
in the same way that the data the pre-trained model was trained on was transformed. This is because the pre-trained 
model has learned to recognize patterns in the data based on the specific transformations applied during its training. 
If your custom data is not transformed in the same way, the pre-trained model may not perform well on it.
"""

"""
8.2.1 Manually create a transform pipeline using torchvision.transforms.Compose and apply it to the dataset.
torchvison.models has pre-trained models
"""
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# resize -> to tensor -> normalize
manual_transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=manual_transform,
    test_transform=manual_transform,
    batch_size=32,
)

print(f"Class names: {class_names}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")

"""
8.2.2 Automatically create - the transforms are defined by the model you want to use, you can get them from torchvision.models.get_model_weights 
and torchvision.models.get_model_transforms.
"""
# get a set of pre-trained weights for a model (EfficientNet_B0 in this case)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()
print(f"Auto transforms: {auto_transforms}")

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=auto_transforms,
    test_transform=auto_transforms,
    batch_size=32,
)
print(f"Class names: {class_names}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")
