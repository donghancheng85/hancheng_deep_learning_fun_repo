import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from common.helper_fucntion import plot_decision_boundary
from lessons.section3_pytorch_workflow.common import plot_prediction

from common.device import get_best_device, print_device_info

# Make device agnostic code
device = get_best_device()
print_device_info(device=device)

"""
6. The missing piece of model here: non-linearity
* What patterns could be drawn if there is an infinite amount of a straight and non-straight lines?
In machine learning terms: an infinite (very large number, but finite of course) of linear and non-linear functions.
"""

"""
6.1 Recreating non-linear data (red and blue circles)
"""

# Make and plot data
n_samples = 1000

X, y = make_circles(
    n_samples=n_samples,
    noise=0.03,
    random_state=42,
)

plt.figure(figsize=(12, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu")
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/c_another_cicle_line_41.png"
)

# Convert data to tensors then to train and test split
X = (
    torch.from_numpy(X).type(torch.float32).to(device)
)  # need torch.float32 because numpy default type is float 64
y = torch.from_numpy(y).type(torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train[:5])
print(y_train[:5])
