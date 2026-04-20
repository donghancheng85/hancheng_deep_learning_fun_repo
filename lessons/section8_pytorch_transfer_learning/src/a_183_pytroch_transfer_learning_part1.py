from zipfile import Path

import torch
import torchvision

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
