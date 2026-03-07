import torch
from torchmetrics import Accuracy
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt
import requests
from pathlib import Path
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
y_blob = torch.from_numpy(y_blob).type(torch.long).to(device)

# 3. Split into and training and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)

X_blob_train: torch.Tensor
X_blob_test: torch.Tensor
y_blob_train: torch.Tensor
y_blob_test: torch.Tensor

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

"""
8.3 Create a loss function and optimizer for multi-class classification model
"""
# Loss function
loss_fn_crossentropy = nn.CrossEntropyLoss()

# Optimizer
optimizer_sdg = torch.optim.SGD(params=blob_model.parameters(), lr=0.1)


"""
8.4 Getting prediction probabilities for a multi-class PyTorch model

In order to evaluate and train and test our model, we need to convert our model's outpu (logits)
to prediction probabilities, then to prediction lables
Logits (raw output of the model) -> prediction probability (use torch.softmax) ->  labels (take the argma of the prediction probabilities)
"""
# Let's the logits of the model
blob_model.eval()
with torch.inference_mode():
    y_logits_before_training: torch.Tensor = blob_model(X_blob_test)

print(f"Shape of logits_before_training is {y_logits_before_training.shape}")

# Conver out model's logts to prediction probabilities
# softmax dim=1 means on dim 1, which represent the class probability sum (row) will be 1
y_prediction_probability_before_training = torch.softmax(
    y_logits_before_training, dim=1
)
print(y_prediction_probability_before_training[:5])

# Conver model prediction probabilities to prediction labels
y_prediction_labels_before_training = torch.argmax(
    torch.softmax(y_logits_before_training, dim=1), dim=1
)
print(
    f"y_prediction_labels_before_training shape is {y_prediction_labels_before_training.shape}"
)

"""
8.5 Create a training loop and testing loop for multi-class classification
"""
# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
epochs = 100

# Data to target device
# X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device) # already done above so no need

# Training/Testing Loop
for epoch in range(epochs):
    ### Training
    blob_model.train()

    # forward
    y_logits_train = blob_model(X_blob_train)
    y_predict_label_train = torch.argmax(torch.softmax(y_logits_train, dim=1), dim=1)

    # calculate the loos
    loss_train: torch.Tensor = loss_fn_crossentropy(y_logits_train, y_blob_train)
    accuracy_train = accuracy_fn(y_true=y_blob_train, y_pred=y_predict_label_train)

    # zero grad
    optimizer_sdg.zero_grad()

    # loss backward
    loss_train.backward()

    # optimizer step
    optimizer_sdg.step()

    ### Testing
    blob_model.eval()
    with torch.inference_mode():
        y_logits_test = blob_model(X_blob_test)
        y_predict_label_test = torch.softmax(y_logits_test, dim=1).argmax(
            dim=1
        )  # another way to get the label

        loss_test: torch.Tensor = loss_fn_crossentropy(y_logits_test, y_blob_test)
        accuracy_test = accuracy_fn(y_true=y_blob_test, y_pred=y_predict_label_test)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Training loss: {loss_train:.5f}, Training accuracy: {accuracy_train:.2f}% | "
                f"Test loss: {loss_test:.5f}, Tesing accuracy: {accuracy_test:.2f}%"
            )

"""
8.6 Making and evaluating predictions with Pytorch multi-class model
"""
# Make predictions
blob_model.eval()
with torch.inference_mode():
    y_logits_after_training = blob_model(X_blob_test)
    y_predict_label_after_training = torch.softmax(
        y_logits_after_training, dim=1
    ).argmax(dim=1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Training Data")
plot_decision_boundary(model=blob_model, X=X_blob_train, y=y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Testing Data")
plot_decision_boundary(model=blob_model, X=X_blob_test, y=y_blob_test)
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/e_line_200_multi_class_classification_results.png"
)


"""
9. A few classification metrics (to evaluate our classification model)

- Accuracy
- Precision
- Recall
- F1 - Score
- Confusion matrix
- Classification report
"""

# Set up metric
torchmetric_accruacy = Accuracy(task="multiclass", num_classes=NUM_CLASS).to(device)

# Calculate accruacy
torchmetric_accruacy_calculated = torchmetric_accruacy(
    y_predict_label_after_training, y_blob_test
)
print(f"Model accuracy is {torchmetric_accruacy_calculated}")
