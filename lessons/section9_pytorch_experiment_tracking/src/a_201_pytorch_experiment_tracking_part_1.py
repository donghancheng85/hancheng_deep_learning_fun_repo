import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from torchinfo import summary

from going_modular.pytorch_project import data_setup, engine, download_data
from common.device import get_best_device, print_device_info

"""
Pytorch experiment tracking is to help you keep track of your experiments, their configurations, and their results. 
This is important because it allows you to compare different experiments, reproduce results, and share your findings with others.
"""

# set up device
device = get_best_device()
print_device_info(device)

# Already exist in going_modular/data/pizza_steak_sushi, so no need to download again
# image_path = download_data.download_data(
#     source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
#     destination="pizza_steak_sushi",
# )
# print(image_path)
