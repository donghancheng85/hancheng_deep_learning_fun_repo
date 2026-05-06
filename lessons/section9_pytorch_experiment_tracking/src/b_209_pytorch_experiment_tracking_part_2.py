import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torchvision import transforms
from datetime import datetime
import os

from torchinfo import summary

from going_modular.pytorch_project import data_setup, engine, download_data
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, create_summary_writer

# set up device
device = get_best_device()
print_device_info(device)

"""
6. Create a function to prepare a SummaryWriter() instance

We want to different models to different folders

One experiment = one folder

We want to track:
 - date/timestamp
 - Experiment name
 - model name
 - hyperparameters
 - training/validation metrics
 - any other relevant information

name will be like:
runs/YYYY-MM-DD-HH-MM-SS/experiment-name/model-name/hyperparameters(extras)
"""

# Example of creating a SummaryWriter instance with the create_summary_writer() function
# example_writer = create_summary_writer(
#     experiment_name="data_10_percent", model_name="efficientnet_b0", extra="5_epochs"
# )
# print(f"Example log directory: {example_writer.log_dir}")

"""
7. Setting up a series of modeling experiments
"""

"""
7.1 What experiments to run? (different hyperparameters)
- Number of epochs
- Number of hidden units
- amount of data to train on
- Learning rate
- different kinds of augmentation
- Choose a different model architecture

This is why transfer learning is so useful, it allows us to quickly iterate through different experiments by only changing a few lines of code.
"""

"""
7.2 What experiments to run in this code? (make it simple currently)
1. Model size - efficientnet_b0 vs efficientnet_b2 (number of parameters - 5.3 million vs 9.2 million)
2. Dataset size - 10% vs 20% (pizza, steak, sushi)
3. Training epochs - 5 vs 10 epochs
"""

"""
7.3 Download the data (already done in part 1, so we can skip this step)
10% dir - going_modular/data/pizza_steak_sushi
20% dir - lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased
"""
