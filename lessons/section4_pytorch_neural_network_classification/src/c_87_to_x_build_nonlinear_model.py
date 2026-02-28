import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from common.helper_fucntion import plot_decision_boundary, accuracy_fn
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
y = torch.from_numpy(y).type(torch.float32).to(device).unsqueeze(dim=1)
print(f"shape of y is {y.shape} | X {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train: torch.Tensor
X_test: torch.Tensor
y_train: torch.Tensor
y_test: torch.Tensor
print(X_train[:5])
print(y_train[:5])

"""
6.2 Building a model with non-linearity
Artificial neural networks are a large combination of linear (straight) and
non-straight (non-linear) functions which are potenitally able to find pattern
in data
"""


# Build a model with non-linear activation functions
class CircleModelV2(nn.Module):
    """
    linear_layer_1 -> ReLu -> linear_layer_2 -> ReLu -> linear_layer_3
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Put the ReLu layer in-between the layers
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model_with_nonlinear = CircleModelV2().to(device)

# Setup loss and optimizer
loss_fn_bcewithlogits = nn.BCEWithLogitsLoss()
optimizer_sdg = torch.optim.SGD(params=model_with_nonlinear.parameters(), lr=0.1)

"""
6.3 Trainin a model with non-linearity
"""
# Randmo seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Epochs of training
epochs = 2000

# Training loop
for epoch in range(epochs):
    # set model to train
    model_with_nonlinear.train()

    # Forward pass
    y_logits_train = model_with_nonlinear(X_train)
    # Note there is difference between torch.sigmoid and nn.Sigmoid
    y_prediction_train = torch.round(torch.sigmoid(y_logits_train))

    # Calculate the loss
    loss_train: torch.Tensor = loss_fn_bcewithlogits(y_logits_train, y_train)
    accuracy_train = accuracy_fn(y_true=y_train, y_pred=y_prediction_train)

    # Zero grad
    optimizer_sdg.zero_grad()

    # loss backward
    loss_train.backward()

    # Optimizer step
    optimizer_sdg.step()

    # Testing
    model_with_nonlinear.eval()
    with torch.inference_mode():
        # Forward pass
        y_logit_test = model_with_nonlinear(X_test)
        y_prediction_test = torch.round(torch.sigmoid(y_logit_test))

        # Calculate the loss
        loss_test: torch.Tensor = loss_fn_bcewithlogits(y_logit_test, y_test)
        accuracy_test = accuracy_fn(y_true=y_test, y_pred=y_prediction_test)

        # Print what is going on
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Training loss: {loss_train:.5f}, Training accuracy: {accuracy_train:.2f}% | "
                f"Test loss: {loss_test:.5f}, Tesing accurady: {accuracy_test:.2f}%"
            )

"""
6.4 Evaluation a model trained with non-linear activation functions
"""
# Make predictions
model_with_nonlinear.eval()
with torch.inference_mode():
    y_prediction_after_train = torch.round(torch.sigmoid(model_with_nonlinear(X_test)))

# Plot the decision boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Data")
plot_decision_boundary(model=model_with_nonlinear, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test Data")
plot_decision_boundary(model=model_with_nonlinear, X=X_test, y=y_test)
# plt.savefig("lessons/section4_pytorch_neural_network_classification/src/c_nonlinear_model_decision_boundary_line_161.png")
