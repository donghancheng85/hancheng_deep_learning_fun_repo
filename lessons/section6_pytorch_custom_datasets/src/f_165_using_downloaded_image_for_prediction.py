import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGG,
)

from typing import Tuple, Dict, List
from pathlib import Path
import pathlib
import os
import random
from PIL import Image
from timeit import default_timer
import matplotlib.pyplot as plt
import requests


device = get_best_device()
print_device_info(device)
"""
Load the model weights from d_161_pytorch_custom_dataset_model_with_augment.py 
and use it to make a prediction on a single image downloaded from the internet.
"""
model_1 = TinyVGG(in_features=3, hidden_units=10, out_features=3).to(device)
_MODEL_PATH = Path("lessons/section6_pytorch_custom_datasets/src/d_model_1.pth")
model_1.load_state_dict(torch.load(_MODEL_PATH, map_location=device, weights_only=True))
model_1.eval()

"""
11. Making a prediction on a custom image downloaded from the internet

Image is not in either the train or test set, but we can still make a prediction on it using our trained model.
"""

# Download an image from the internet and save it to disk (done manually in this case)
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
custom_image_path = data_path / "04-pizza-dad.jpg"

if not custom_image_path.exists():
    with open(custom_image_path, "wb") as f:
        f.write(
            requests.get(
                "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
            ).content
        )
        print(f"Custom image downloaded and saved to: {custom_image_path}")
else:
    print(f"Custom image already exists at: {custom_image_path}")
