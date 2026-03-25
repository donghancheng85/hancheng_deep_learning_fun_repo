import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable

from timeit import default_timer
import time
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import torchmetrics

import matplotlib.pyplot as plt

from common.helper_fucntion import accuracy_fn, print_train_time
from common.device import get_best_device, print_device_info
from lessons.section5_pytorch_computer_vision.common.common import (
    evaluate_model,
    train_step,
    test_step,
)

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
BATCH_SIZE = 32

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
7. Model 2: Building a Convolutional Neural Network (CNN)

CNN also called ConvNets
CNN are know for their capabilities to find patterns in visual data

Resource: https://poloclub.github.io/cnn-explainer/
"""


# Create a Convolutional Neural Network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture that replicate the Tiny VGG (Visual Geometry Group)
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        # --- Conv Block 1 ---
        # Input tensor shape: [batch, input_shape, 28, 28]  (e.g. [32, 1, 28, 28] for FashionMNIST)
        #
        # Conv2d spatial formula: H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
        #
        # Conv2d #1:  H_out = floor((28 + 2*1 - 3) / 1 + 1) = 28  →  [batch, hidden_units, 28, 28]
        # ReLU: no shape change, applies element-wise non-linearity
        # Conv2d #2:  H_out = 28 (same padding=1, kernel=3, stride=1) →  [batch, hidden_units, 28, 28]
        # ReLU: no shape change
        # MaxPool2d (kernel=2, stride defaults to kernel=2):
        #             H_out = floor(28 / 2) = 14   →  [batch, hidden_units, 14, 14]
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,  # 1 for grayscale (FashionMNIST), 3 for RGB
                out_channels=hidden_units,  # number of learned filters / feature maps
                kernel_size=3,  # 3×3 sliding window
                stride=1,  # move window 1 pixel at a time
                padding=1,  # pad borders so spatial size is preserved after conv
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,  # halves spatial dimensions: 28×28 → 14×14
            ),
        )

        # --- Conv Block 2 ---
        # Input: [batch, hidden_units, 14, 14]
        # Conv2d #1 & #2 (padding=1, kernel=3, stride=1): shape unchanged → [batch, hidden_units, 14, 14]
        # MaxPool2d (kernel=2): 14×14 → 7×7  →  [batch, hidden_units, 7, 7]
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14×14 → 7×7
        )

        # --- Classifier (fully-connected head) ---
        # Flatten: [batch, hidden_units, 7, 7] → [batch, hidden_units * 7 * 7]
        #   in_features = hidden_units * 7 * 7
        #     - hidden_units: number of feature maps coming out of the last conv block
        #     - 7 * 7:        spatial size after two MaxPool2d(kernel=2) on a 28×28 input
        #                     (28 → 14 → 7)
        # Linear: maps the flattened vector to class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 7 * 7,  # must match flattened conv output
                out_features=output_shape,  # one logit per class
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28]
        x = self.conv_block_1(x)  # → [batch, hidden_units, 14, 14]
        # print(
        #     f"Output shape of conv_block_1 {x.shape}"
        # )  # trick to check what will be the shape of the output tensor of CNN
        x = self.conv_block_2(x)  # → [batch, hidden_units,  7,  7]
        # print(f"Output shape of conv_block_1  {x.shape}")
        x = self.classifier(x)  # → [batch, output_shape]
        # print(f"Output shape of classifier  {x.shape}")
        return x


torch.manual_seed(42)

model_2 = FashionMNISTModelV2(
    input_shape=1,  # number of color channels in the image, FashionMNIST has only 1 channel
    hidden_units=10,
    output_shape=len(class_name),
).to(device)


"""
7.1 Stepping through nn.Conv2d()
"""
torch.manual_seed(42)

# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64))  # batch, color channel, hight, width
test_image = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
# print(f"Test image: \n{test_image}")

# Create a Conv2d layer
conv_layer = nn.Conv2d(
    in_channels=3,
    out_channels=10,
    kernel_size=3,
    stride=1,
    padding=0,
)

# Pass the data throught the convolutional layer
conv_output = conv_layer(test_image)
print(f"after the conv_layer, conv_output shape is {conv_output.shape}")

"""
7.2 Stepping throught nn.MaxPool2d()
"""
# Use the created "simulation" images to pass through the MaxPool2d
print(f"\nImage batch shape: {images.shape}")
print(f"Single image shape: {test_image.shape}")
print(
    f"Single image shape with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}"
)

# Create nn.MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv layer
test_image_through_conv = conv_layer(test_image)
print(f"test image through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(
    f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}"
)

# Create a smaller tensor so easy to visualize, with similar tensor
torch.manual_seed(42)
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"\nrandom_tensor: \n{random_tensor}")
print(f"random_tensor shape: {random_tensor.shape}")

# Create a nother MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass the random tensor throught the max pool layer
max_pool_tensor: torch.Tensor = max_pool_layer(random_tensor)

print(f"\n max_pool_tensor: \n {max_pool_tensor}")
print(f"max_pool_tensor shape: {max_pool_tensor.shape}")

# To check how the model change dim of the image, pass image through model_2
out_image = model_2(image.unsqueeze(dim=0).to(device))
print(f"out_image shape is {out_image.shape}")

rand_image_tensor = torch.randn(size=(1, 1, 28, 28))
print(f"rand image tensor shape: {rand_image_tensor.shape}")
out_rand_image_tensor = model_2(rand_image_tensor.to(device))
print(f"out_rand_image_tensor shape: {out_rand_image_tensor.shape}")

"""
7.3 setup loss funcion and optimizer for model_2 (TinyVGG)
"""
# Setup loss function/eval metrics/optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

"""
7.4 Trainig and testing model_2 (TinyVGG)
"""
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Measure time
train_time_start_model_2 = default_timer()

# Train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n-----------")
    train_step(
        model=model_2,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accruacy_fn=accuracy_fn,
        device=device,
    )

    test_step(
        model=model_2,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )

train_time_end_model_2 = default_timer()

total_train_time_model_2 = print_train_time(
    start=train_time_start_model_2,
    end=train_time_end_model_2,
    device=device,
)


# Get model_2 results
model_2_result = evaluate_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)
print(model_2_result)
# Sample output: {'model_name': 'FashionMNISTModelV2', 'model_loss': 0.3200104832649231, 'model_accuracy': 88.29872204472844}

"""
8. Load saved model_0 and model_1 and compare them with model_2

Both model_0 and model_1 were saved as state_dicts via torch.save(model.state_dict(), path).
To reload a state_dict you must:
  1. Re-create the model instance with the same architecture / constructor args.
  2. Call model.load_state_dict(torch.load(path, weights_only=True)).
  3. Move the model to the target device with model.to(device).
  4. Call model.eval() so dropout/batchnorm behave correctly during inference.
"""


# --- Re-define model_0 architecture (from section5/src/a_103_...) ---
# Baseline: Flatten → Linear → Linear  (no activation, no conv)
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


# --- Re-define model_1 architecture (from section5/src/b_112_...) ---
# Improved baseline: adds ReLU after each Linear layer
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


# --- Paths to saved state dicts ---
MODEL_PATH = Path("lessons/section5_pytorch_computer_vision/models")
MODEL_0_PATH = MODEL_PATH / "section5_model_0_fashionMNIST.pth"
MODEL_1_PATH = MODEL_PATH / "section5_model_1_fashionMNIST.pth"

# --- Instantiate with the same constructor args used during training ---
# FashionMNIST images are 28×28 = 784 pixels; 10 classes
model_0 = FashionMNISTModelV0(
    input_shape=784, hidden_units=10, output_shape=len(class_name)
)
model_1 = FashionMNISTModelV1(
    input_shape=784, hidden_units=10, output_shape=len(class_name)
)

# --- Load state dicts ---
# weights_only=True is recommended (avoids arbitrary code execution from pickle)
model_0.load_state_dict(torch.load(f=MODEL_0_PATH, weights_only=True))
model_1.load_state_dict(torch.load(f=MODEL_1_PATH, weights_only=True))

# Move to device and switch to eval mode
model_0.to(device)
model_1.to(device)

# --- Evaluate loaded models on the test set ---
model_0_result = evaluate_model(
    model=model_0,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)

model_1_result = evaluate_model(
    model=model_1,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)

# --- Side-by-side comparison ---
compare_results = pd.DataFrame([model_0_result, model_1_result, model_2_result])
compare_results["training_time"] = [
    4.090,
    3.087,
    total_train_time_model_2,
]  # model_0 1 time is copied from other files
print("\nModel comparison:")
print(compare_results)
"""
Sample output:
Model comparison:
            model_name  model_loss  model_accuracy  training_time
0  FashionMNISTModelV0    0.476639       83.426518        4.09000
1  FashionMNISTModelV1    0.685001       75.019968        3.08700
2  FashionMNISTModelV2    0.334073       88.238818        4.37878
"""

# Visualize out model results
compare_results.set_index("model_name")["model_accuracy"].plot(kind="barh")
# plt.figure(figsize=(12, 12))
plt.xlabel("accuracy %")
plt.ylabel("model")
plt.tight_layout()  # auto-adjust margins so y-axis model name labels are not clipped
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/c_line_466_visualize_model_comparsion.png"
)

"""
9. Save model_2 so it can be loaded for future comparisons

We only save the state_dict (learned weights), not the full model object.
Reasons:
  - Smaller file size (no Python class code stored).
  - More portable: works across PyTorch versions as long as the class definition exists.
  - To reload: instantiate FashionMNISTModelV2 with the same args, then call load_state_dict().
"""
MODEL_2_SAVE_PATH = MODEL_PATH / "section5_model_2_fashionMNIST.pth"
# MODEL_PATH already defined above as Path("lessons/section5_pytorch_computer_vision/models")
MODEL_PATH.mkdir(
    parents=True, exist_ok=True
)  # create the directory if it doesn't exist yet
torch.save(obj=model_2.state_dict(), f=MODEL_2_SAVE_PATH)
print(f"Model 2 saved to: {MODEL_2_SAVE_PATH}")
