import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from common.helper_fucntion import plot_decision_boundary, accuracy_fn
from common.helper_fucntion import plot_decision_boundary, accuracy_fn
from common.device import get_best_device, print_device_info

device = get_best_device()
print_device_info(device)

"""
8. Multi-class classification problem
Multi-class classification -- more than two classes
"""

"""
8.1 Creating a toy multi-class dataset
"""
# set the hyperparmeters for data creation
NUM_CLASS = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(
    n_samples=1000,
    n_features=NUM_FEATURES,
    centers=NUM_CLASS,
    cluster_std=1.5,
    random_state=RANDOM_SEED,
)

# 2. turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32).to(device)
y_blob = torch.from_numpy(y_blob).type(torch.float32).to(device).unsqueeze(dim=1)

# 3. Split into and training and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)

X_blob_train: torch.Tensor
X_blob_test: torch.Tensor
y_blob_train: torch.Tensor
y_blob_train: torch.Tensor

# 4. visualize the data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0].cpu(), X_blob[:, 1].cpu(), c=y_blob.cpu(), cmap="RdYlBu")
plt.grid()
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/e_line_57_multiclass_origin_data.png"
)

"""
8.2 Building a multi-class classification model in PyTorch
"""


# Build a multi-class classifiction model
class BlobModel(nn.Module):
    def __init__(
        self, input_features: int, out_features: int, hidden_units: int = 8
    ) -> None:
        """
        Ctor of class BlobModel (a multi-class classificatio model)
        out_features - output classes
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.linear_layer_stack(x)


# Create a instance
blob_model = BlobModel(
    input_features=NUM_FEATURES, out_features=NUM_CLASS, hidden_units=8
).to(device)

print(blob_model.state_dict())
