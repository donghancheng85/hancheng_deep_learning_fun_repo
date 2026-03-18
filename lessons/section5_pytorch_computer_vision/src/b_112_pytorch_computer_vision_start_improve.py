import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable

from timeit import default_timer
import time
from tqdm.auto import tqdm
from pathlib import Path

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import torchmetrics

import matplotlib.pyplot as plt

from common.helper_fucntion import accuracy_fn, print_train_time
from common.device import get_best_device, print_device_info
from lessons.section5_pytorch_computer_vision.common.common import evaluate_model

"""
0. Computer vision libraries in PyTorch
- torchvision - base domain for computer vision in PyTorch
- torchvision.datasets - get datasets and data loading for computer vision
- torchvision.models - get pretrained computer vision models that you can leverage for your own problems
- torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
- torch.utils.data.Dataset - Base dataset class for PyTorch
- torch.utils,data.Dataload - creates a Python iterable over a dataset
"""

"""
1. Getting a dataset
Dataset will be used is FashionMNIST from torchvision.datasets
"""
# Set up training data
train_data = datasets.FashionMNIST(
    root="lessons/section5_pytorch_computer_vision/data",  # where to download data to
    train=True,  # do we want the training dataset?
    download=False,  # do we want to download
    transform=ToTensor(),  # how do we want to transform the data
    target_transform=None,  # how do we want to transform the labels/targets
)

test_data = datasets.FashionMNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None,
)

# Check sample size
print(f"Train data length {len(train_data)} | test data {len(test_data)}")

# See the first training example
image, label = train_data[0]
image: torch.Tensor
class_name = train_data.classes
"""
2. Prepare DataLoader

Currently, FashionMNIST is in the form of PyTorch Datasets
DataLoader turns datasets to a Python iterable
Specifically, we want to turn our data into batches or mini-batches
Why batches? 
1. It is more computationally efficiently, as in , your computing hardware may not be able to load 60000 images in one hit.
So we break it down to 32 images a time (batch size of 32 -- can be changed)
2. It gives our nerual network more chances to update its gradients per epoch
"""
# Set up the batch size hyperparameter
# Larger batch size keeps the GPU fed and reduces per-batch kernel launch overhead
BATCH_SIZE = 256

# Turn data set into iterables (batches)
# pin_memory=True: pre-allocates data in page-locked memory for faster CPU->GPU transfer
# num_workers=4: loads batches in parallel background processes so GPU never waits for data
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

# No need to shuffle test data, it will not be used in training
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

# Check out the train and test dataloader
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(
    f"Length of train_dataloader: {len(train_dataloader)} | test_dataloader {len(test_dataloader)}"
)
print(
    f"Batch size: train_dataloader {train_dataloader.batch_size} | test_dataloader {test_dataloader.batch_size}"
)

# Checkout what is inside train_dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch: torch.Tensor
train_labels_batch: torch.Tensor

print(
    f"one batch of train_features {train_features_batch.shape} | train_labels {train_labels_batch.shape}"
)
# one batch of train_features torch.Size([32, 1, 28, 28]) | train_labels torch.Size([32])

"""
5. Set up device agnostic-code (if GPU is avaliable)
"""
device = get_best_device()
print_device_info(device=device)


"""
6. Model 1, building a better model with non-linearity

Non-linearity is powerful when dealing with problem with non-linearity.
"""


# Create a model with non-linear and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # Flatten input into a single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


# Create an instance of model_1, input shape is 28*28
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape=784, hidden_units=10, output_shape=len(class_name)
).to(device)

print(f"model_1 is on {next(model_1.parameters()).device}")


"""
6.1 Setup loss, optimizer and evluation metric
"""

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model_1.parameters(),
    lr=0.1,
)

"""
6.2 Functionizing training and evaluation/testing loop

Will create function for:
- training loop - train_step()
- testing loop - test_step()
"""


def train_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accruacy_fn,
    device: torch.device = device,
):
    """
    Perform a trining with model trying to learn on data_loader
    """
    train_loss, train_accuracy = 0, 0

    # Put model into training mode and target device
    model.train()

    # A a loop to loop through the taining batches
    for batch, (X, y) in enumerate(data_loader):
        # put data on target device
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_logits_train = model(X)

        # 2. calculate the loss accruacy (per batch)
        loss_train_batch: torch.Tensor = loss_fn(y_logits_train, y)
        train_loss += loss_train_batch  # accumulate the training loss so we can calculate the average loss of the batches
        train_accuracy += accruacy_fn(y_true=y, y_pred=y_logits_train.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss_train_batch.backward()

        # 5. optimizer step
        optimizer.step()  # model parameter will be updated once per batch

    # Divide total train loss and accuracy by length of data_loader
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}%")


def test_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    """
    Perform a testing loop step on model going over data_loader
    """
    test_loss, test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # 1. Forward pass
            y_logits_test: torch.Tensor = model(X_test)

            # 2. Calculate the loos (accumulate)
            loss_test_batch = loss_fn(y_logits_test, y_test)
            test_loss += loss_test_batch

            # 3. calculate the accuracy
            test_accuracy += accuracy_fn(
                y_true=y_test, y_pred=y_logits_test.argmax(dim=1)
            )

        # Calculate the test loss average per batch
        test_loss /= len(data_loader)

        # Calculate the test accuracy aversge per batch
        test_accuracy /= len(data_loader)

        print(f"\nTest loss: {test_loss:.5f}, Test accuracy: {test_accuracy:.2f}%\n")


torch.manual_seed(42)

# Measure time (on GPU)
train_time_start_on_gpu = default_timer()

# Set the number of epochs (keep is small for faster training time)
epochs = 6

# Create a optimization and evaluation loop uisng train_step() and test_step()
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n ------")

    train_step(
        model=model_1,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accruacy_fn=accuracy_fn,
        device=device,
    )

    test_step(
        model=model_1,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )

train_time_end_on_gpu = default_timer()

total_train_time_on_gpu = print_train_time(
    start=train_time_start_on_gpu,
    end=train_time_end_on_gpu,
    device=str(next(model_1.parameters()).device),
)
# Sample output: Train time on cuda:0: 2.445 seconds
# ^ increased batch size to fully utilize GPU so accuracy will be lower but time is better.

"""
Note: Sometimes, depending on data/hardware, the model may train faster on CPU than GPU.
Reason:
1. the overhead of copying data/model to and from GPU out weight the comput benefits offered by GPU.
2. The CPU may have a better performance than GPU (rare)
"""

# Get model_1 results dictionary
model_1_results = evaluate_model(
    model=model_1,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)

print(model_1_results)
# Sample output: {'model_name': 'FashionMNISTModelV1', 'model_loss': 0.9154605269432068, 'model_accuracy': 66.19140625}

"""
Save model_1 for future comparison
"""

MODEL_PATH = Path("lessons/section5_pytorch_computer_vision/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_PATH / "section5_model_1_fashionMNIST.pth"
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
