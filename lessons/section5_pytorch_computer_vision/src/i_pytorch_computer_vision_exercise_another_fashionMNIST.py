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

from common.device import get_best_device, print_device_info

from lessons.section5_pytorch_computer_vision.common.common import (
    train_step,
    test_step,
    evaluate_model,
)
from common.helper_fucntion import accuracy_fn, print_train_time

"""
13. Use a model similar to the trained model_2 from this notebook to make predictions on the test torchvision.datasets.FashionMNIST dataset.
    Then plot some predictions where the model was wrong alongside what the label of the image should've been.
    After visualizing these predictions do you think it's more of a modelling error or a data error?
    As in, could the model do better or are the labels of the data too close to each other (e.g. a "Shirt" label is too close to "T-shirt/top")?
"""

# Load data
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

class_name = train_data.classes
print(f"class names are: {class_name}")

# Convert to Dataloader
BATCH_SIZE = 32
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


# Recreate the TinyVGG model
class TinyVGG(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_features: int) -> None:
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
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
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_stack_2 = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=2),
        )

        # --- Calculate in_features dynamically via a dummy forward pass ---
        # Best practice: instead of hardcoding the flattened size (which breaks
        # whenever you change kernel_size, padding, or input resolution),
        # run a zero tensor through the conv stacks to let PyTorch compute the
        # output shape for us automatically.
        # torch.no_grad(): skip gradient tracking — this is just a shape probe.
        with torch.no_grad():
            dummy = torch.zeros(1, in_features, 28, 28)  # [batch=1, C, H, W]
            dummy = self.conv_stack_2(self.conv_stack_1(dummy))
            # dummy shape after both conv stacks: [1, hidden_units, H_out, W_out]
            linear_in_features = dummy.flatten(start_dim=1).shape[1]
            # flatten(start_dim=1): collapse all dims except batch → [1, hidden_units*H_out*W_out]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=linear_in_features,  # auto-computed above
                out_features=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_stack_2(self.conv_stack_1(x)))


device = get_best_device()
print_device_info(device)

my_tinyvgg_fashionmnist_gpu = TinyVGG(
    in_features=1,
    hidden_units=10,
    out_features=len(class_name),
).to(device=device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=my_tinyvgg_fashionmnist_gpu.parameters(),
    lr=0.1,
)

EPOCHS = 6
train_start_time = default_timer()
for epoch in tqdm(range(EPOCHS)):
    train_step(
        model=my_tinyvgg_fashionmnist_gpu,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accruacy_fn=accuracy_fn,
        device=device,
    )

    test_step(
        model=my_tinyvgg_fashionmnist_gpu,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )
train_end_time = default_timer()

print_train_time(start=train_start_time, end=train_end_time, device=device)

my_tinyvgg_fashionmnist_gpu_result = evaluate_model(
    model=my_tinyvgg_fashionmnist_gpu,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)
print(my_tinyvgg_fashionmnist_gpu_result)

# Collect all predictions and true labels across the entire test set
all_pred_labels: list[torch.Tensor] = []
all_true_labels: list[torch.Tensor] = []

my_tinyvgg_fashionmnist_gpu.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logits = my_tinyvgg_fashionmnist_gpu(X)
        all_pred_labels.append(y_logits.argmax(dim=1).cpu())
        all_true_labels.append(y.cpu())

pred_labels_tensor = torch.cat(all_pred_labels, dim=0)  # [10000]
true_labels_tensor = torch.cat(all_true_labels, dim=0)  # [10000]

# Find the indexes where prediction is wrong
# wrong_indexes: 1D tensor of indexes into test_data where model made a mistake
wrong_indexes = (pred_labels_tensor != true_labels_tensor).nonzero(as_tuple=True)[0]
print(f"Total wrong predictions: {len(wrong_indexes)} / {len(test_data)}")
print(f"First 10 wrong indexes: {wrong_indexes[:10]}")

# Plot prediction and compare with origin
plt.figure(figsize=(12, 12))
nrows = 5
ncolunms = 5
for index in range(nrows * ncolunms):
    plt.subplot(nrows, ncolunms, index + 1)

    # get a random position into wrong_indexes, then look up the actual test_data index
    rand_pos = torch.randint(low=0, high=len(wrong_indexes), size=(1,)).item()
    actual_idx = wrong_indexes[int(rand_pos)].item()  # index into test_data

    raw_image, raw_label = test_data[int(actual_idx)]
    prediction_label = pred_labels_tensor[
        int(actual_idx)
    ]  # predicted class label (int)

    true_class_name = class_name[raw_label]
    prediction_class_name = class_name[prediction_label]

    plt.imshow(
        raw_image.squeeze(dim=0), cmap="gray"
    )  # remove the first dim -> (1, 28, 28)
    title_text = f"Pred: {prediction_class_name} | Truth: {true_class_name}"

    if prediction_label == raw_label:
        plt.title(title_text, fontsize=10, c="g")  # correct prediction -- green
    else:
        plt.title(title_text, fontsize=10, c="r")  # wrong prediction -- red
    plt.axis(False)

plt.tight_layout()
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/i_line_241_visualize_random_wrong_predictions.png"
)
