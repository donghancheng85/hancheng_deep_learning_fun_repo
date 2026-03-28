import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from timeit import default_timer
from tqdm.auto import tqdm
import pathlib

import matplotlib.pyplot as plt

from lessons.section5_pytorch_computer_vision.common.common import (
    train_step,
    test_step,
    evaluate_model,
)
from common.helper_fucntion import accuracy_fn, print_train_time
from common.device import get_best_device, print_device_info

"""
4. Load the torchvision.datasets.MNIST() train and test datasets.
"""
train_data = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=None,  # labels are plain integers: e.g. 7
)

test_data = datasets.MNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=None,
)

class_name = train_data.classes
print(f"class names are: {class_name}")

"""
7. Turn the MNIST train and test datasets into dataloaders 
   using torch.utils.data.DataLoader, set the batch_size=32.
"""
BATCH_SIZE = 32
# dataloader is an iterable over the dataset, it will yield batches of data when iterated over
# each element of the dataloader is a list of two elements: [batch of images, batch of labels]
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

"""
8. Recreate model_2 used in this notebook (the same model from the CNN Explainer website, 
   also known as TinyVGG) capable of fitting on the MNIST dataset.
"""


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


"""
9. Train the model you built in exercise 8. 
   on CPU and GPU and see how long it takes on each.
"""
# 9.1 train model on GPU
device = get_best_device()
print_device_info(device)

my_tinyvgg_gpu = TinyVGG(
    in_features=1,
    hidden_units=10,
    out_features=len(class_name),
).to(device=device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=my_tinyvgg_gpu.parameters(),
    lr=0.1,
)

EPOCHS = 3
train_start_time = default_timer()
for epoch in tqdm(range(EPOCHS)):
    train_step(
        model=my_tinyvgg_gpu,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accruacy_fn=accuracy_fn,
        device=device,
    )

    test_step(
        model=my_tinyvgg_gpu,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )
train_end_time = default_timer()

print_train_time(start=train_start_time, end=train_end_time, device=device)

my_tinyvgg_gpu_result = evaluate_model(
    model=my_tinyvgg_gpu,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device,
)
print(my_tinyvgg_gpu_result)

"""
Sample output:
Train time on cuda: 6.061 seconds
{'model_name': 'TinyVGG', 'model_loss': 0.04290688410401344, 'model_accuracy': 98.58226837060703}
"""

"""
Save model for inference
"""

MODEL_PATH = pathlib.Path("lessons/section5_pytorch_computer_vision/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_PATH / "section5_model_tinyvgg_mnist.pth"

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=my_tinyvgg_gpu.state_dict(), f=MODEL_SAVE_PATH)
print("Model saved!")
