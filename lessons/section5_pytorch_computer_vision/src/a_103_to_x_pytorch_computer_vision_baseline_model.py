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

from common.helper_fucntion import accuracy_fn

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
print(f"Class names {train_data.classes}")
print(f"Class to index {train_data.class_to_idx}")
print(
    f"Type of image {type(image)} | shape {image.shape} -> [C, H, W] | label class {class_name[label]}"
)

"""
1.2 Visualizing the data
"""
plt.imshow(image.squeeze(dim=0), cmap="gray")
plt.title(class_name[label])
plt.axis(False)
plt.savefig("lessons/section5_pytorch_computer_vision/src/a_line_55_visualize_data.png")

# plot more random images, so we have a better idea what the data contains
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_index = torch.randint(low=0, high=len(train_data), size=(1,)).item()
    img, label = train_data[int(random_index)]
    img: torch.Tensor
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(dim=0), cmap="gray")
    plt.title(class_name[label])
    plt.axis(False)
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/a_line_72_visualize_random_train_data.png"
)

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
BATCH_SIZE = 32

# Turn data set into iterables (batches)
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# No need to shuffle test data, it will not be used in training
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Check out the train and test dataloader
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(
    f"Length of train_dataloader: {len(train_dataloader)} | test_dataloader {len(test_dataloader)}"
)
print(
    f"Batch size: train_dataloader {train_dataloader.batch_size} | test_dataloader {test_dataloader.batch_size}"
)

# Checkout what is inside train_dateloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch: torch.Tensor
train_labels_batch: torch.Tensor

print(
    f"one batch of train_features {train_features_batch.shape} | train_labels {train_labels_batch.shape}"
)
# one batch of train_features torch.Size([32, 1, 28, 28]) | train_labels torch.Size([32])

# Visual a sample (how to interact with dataloader)
torch.manual_seed(42)
random_index = torch.randint(low=0, high=len(train_features_batch), size=(1,)).item()
img_batch, label_batch = (
    train_features_batch[int(random_index)],
    train_labels_batch[int(random_index)],
)
plt.figure()
plt.imshow(img_batch.squeeze(dim=0), cmap="gray")
plt.title(class_name[label_batch])
plt.axis(False)
print(f"Image size: {img_batch.shape}")
print(f"Label: {label}, label size {label_batch.shape}")
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/a_line_128_visualize_img_in_train_dataloader.png"
)

"""
3. Model 0: buid a baseline model

When starting to build a series of machine learning modeling experiments, it's best practice is to 
start with a baseline model

A baseline mode is a simple model you will try and improve upon with subsequent model/experiments.

In other words: start simply and add complexity when necessary
"""

# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]
print(f"Shape of x is {x.shape} -> [color_channel, height, weight]")
# Shape of x is torch.Size([1, 28, 28]) -> [color_channel, height, weight]

# Flatten the sample
x_flatten: torch.Tensor = flatten_model(x)
print(
    f"After Flatten, x_flatten shape is {x_flatten.shape} -> [color_channel, height*weight]"
)
# After Flatten, x_flatten shape is torch.Size([1, 784]) -> [color_channel, height*weight]


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


torch.manual_seed(42)

# Setup model with input parameter
model_0 = FashionMNISTModelV0(
    input_shape=784,  # 28*28
    hidden_units=10,
    output_shape=len(class_name),  # one for every class
).to("cpu")

# check what does the model do
dummy_x = torch.rand([1, 1, 28, 28])
dummy_y = model_0(dummy_x)
print(f"dummy_y is {dummy_y} | shape is {dummy_y.shape}")

"""
3.1 Set up loss, optimizer and evaluation metrics

* Loss function - this is a multi-class classification problem so the loss
function will be nn.CrossEntropyLoss()

* optimizer - the optimzer torch.optim.SGD (stochastic gradient descent)

* Evaluation metric - this is a classification problem, we will use accruacy as
evaluation metric
"""
# Metric functions
accuracy_calculator = torchmetrics.Accuracy(
    task="multiclass", num_classes=len(class_name)
).to("cpu")

# Loss function and optimzer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model_0.parameters(),
    lr=0.1,
)

"""
3.2 Create a function to time our experiments

Machine learning is very experimental.

Two main things we often want to track are:
1. Model's metrics (loss, accuracy, precision, recall...)
2. How fast it runs
The above two sometimes are trade-offs
"""


def print_train_time(start: float, end: float, device: torch.device | str) -> float:
    """
    Print the difference between start and end time
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# test print_train_time
start_time = default_timer()
time.sleep(1.1)
end_time = default_timer()
print_train_time(start=start_time, end=end_time, device="cpu")

"""
3.3 Creating a training loop and training a model on batches of data

1. Loop through epochs
2. Loop through training batches, perfrom training steps, calculate train loss per batch
3. Loop through testing batches, perfrom testing steps, calculate test loss per batch
4. Print out what's happening
5. Time it all (how long the training takes)
"""

# set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = default_timer()

# Set the number of epochs (keep is small for faster training time)
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n ------")
    ### Training
    train_loss = 0

    # A a loop to loop through the taining batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()

        # 1. forward pass
        y_logits_train = model_0(X)

        # 2. calculate the loss (per batch)
        loss_train_batch: torch.Tensor = loss_fn(y_logits_train, y)
        train_loss += loss_train_batch  # accumulate the training loss so we can calculate the average loss of the batches

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss_train_batch.backward()

        # 5. optimizer step
        optimizer.step()  # model parameter will be updated once per batch

        # Print out stuff
        if batch % 400 == 0:
            print(
                f"Looked at: {batch * len(X)} / {len(train_data)}  samples "
            )  # Use train_data here since it is in the "dataset" field of train_dataloader

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    ### Testing
    test_loss, test_accuracy = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward pass
            y_logits_test: torch.Tensor = model_0(X_test)

            # 2. Calculate the loos (accumulate)
            loss_test_batch = loss_fn(y_logits_test, y_test)
            test_loss += loss_test_batch

            # 3. calculate the accuracy
            test_accuracy += accuracy_fn(
                y_true=y_test, y_pred=y_logits_test.argmax(dim=1)
            )

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test accuracy aversge per batch
        test_accuracy /= len(test_dataloader)

    # Print out
    print(
        f"\n Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}"
    )

# Calucate the training time
train_time_end_on_cpu = default_timer()
total_train_time_model_0_cpu = print_train_time(
    start=train_time_start_on_cpu,
    end=train_time_end_on_cpu,
    device=str(next(model_0.parameters()).device),
)
# Sample output: Train time on cpu: 3.966 seconds

"""
4. Make prediction and get model_0 result (evaluation)
"""
torch.manual_seed(42)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
) -> dict[str, str | float]:
    """
    Returns a dictionary containing the results of model predicting on data_loader
    """
    loss, accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_logits_prediction: torch.Tensor = model(X)

            # Accmulate the loss and accruacy values per batch
            loss_batch = loss_fn(y_logits_prediction, y)
            loss += loss_batch
            accuracy_batch = accuracy_fn(y, y_logits_prediction.argmax(dim=1))
            accuracy += accuracy_batch

        # Scale loss and accuracy to find the average loss/acc per batch
        loss /= len(data_loader)
        accuracy /= len(data_loader)
        loss: torch.Tensor

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_accuracy": accuracy,
    }

# Calculate model_0 results on test dataset
model_0_results = evaluate_model(
    model=model_0,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
)

print(model_0_results)

"""
4.1. Save model_0 for future comparison
"""
MODEL_PATH = Path("lessons/section5_pytorch_computer_vision/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_PATH / "section5_model_0_fashionMNIST.pth"
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
