import torch
from torch import nn
import matplotlib.pyplot as plt
from common.device import get_best_device, print_device_info

# check pytroch version
print(f"current Pytroch version is {torch.__version__}")

# Device agnostic code (uisng stuff in common)
best_device = get_best_device()
print_device_info(best_device)
