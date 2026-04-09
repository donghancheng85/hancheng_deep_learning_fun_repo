import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import torchinfo
import json

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
)

from typing import Tuple, Dict, List
from pathlib import Path
import pathlib
import os
import random
from PIL import Image
from timeit import default_timer
import matplotlib.pyplot as plt

"""
0. Setting up device-agnostic code
"""
device = get_best_device()
print_device_info(device)

"""
1. Setup data paths
   (assumes pizza_steak_sushi data already downloaded and unzipped by a_135_to_x_*.py)
"""
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"


"""
2. Helper function to get class names from directory structure
"""


def find_classes(directory: str | Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        directory (str | Path): The root directory of the dataset.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple of class names and a dict mapping class name -> index.
    """
    class_names_found = sorted(
        [entry.name for entry in os.scandir(directory) if entry.is_dir()]
    )
    if not class_names_found:
        raise FileNotFoundError(f"Couldn't find any class folders in {directory}.")
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names_found)}
    return class_names_found, class_to_idx


"""
3. Custom dataset class (replicates torchvision.datasets.ImageFolder)
"""


class ImageFolderCustom(Dataset):
    def __init__(self, target_dir: str | Path, transform: v2.Compose = None) -> None:
        self.paths = list(pathlib.Path(target_dir).rglob("*/*.jpg"))
        self.transform = transform
        self.class_names, self.class_to_idx = find_classes(target_dir)

    def load_images(self, index: int) -> Image.Image:
        return Image.open(self.paths[index])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor | Image.Image, int]:
        image = self.load_images(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, class_idx


"""
7. Mode 0: TinyVGG architecture (same as section 5) without augmentation
"""

"""
7.1 Creating transforms for training and testing (no augmentation)
"""
# Create simple transform
simple_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

"""
7.2 Loading image data using ImageFolder with simple_transform
"""
train_data_simple = datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform,
)
test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform,
)

class_names = train_data_simple.classes
print(f"Class names: {class_names}")
print(f"Number of training samples: {len(train_data_simple)}")
print(f"Number of testing samples: {len(test_data_simple)}")

# 2. Create Dataloader
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Using {NUM_WORKERS} workers for dataloader.")
train_dataloader_simple = DataLoader(
    dataset=train_data_simple,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
test_dataloader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


"""
7.3 Re-define TinyVGG architecture (same as section 5)
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
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Calculate in_features dynamically via a dummy forward pass ---
        # Best practice: instead of hardcoding the flattened size (which breaks
        # whenever you change kernel_size, padding, or input resolution),
        # run a zero tensor through the conv stacks to let PyTorch compute the
        # output shape for us automatically.
        # torch.no_grad(): skip gradient tracking — this is just a shape probe.
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_features, 64, 64
            )  # [batch=1, C, H, W] — matches Resize(64,64) in transform
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


torch.manual_seed(42)
model_0 = TinyVGG(in_features=3, hidden_units=10, out_features=len(class_names)).to(
    device
)
print(model_0)

"""
7.4 Use torchinfo to check the model summary and output shapes at each layer
    (note: torchinfo is a third-party library, install via pip install torchinfo)
"""
torchinfo.summary(model_0, input_size=(1, 3, 64, 64))

"""
7.5 Create train and test step functions, can be found in common/common.py under train_step() and test_step()
"""
"""
7.6 Create a train function to combine train_step() and test_step() for multiple epochs, can be found in common/common.py under train()
"""

"""
7.7 Train and evaluate model_0 on the simple dataset (no augmentation)
"""
# set the random seeds for reproducibility, note: just for testing purposes, not best practice for real training
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set epochs and optimizer
_NUM_EPOCHS = 10

# Recreate an instance of TinyVGG
model_0 = TinyVGG(
    in_features=3, hidden_units=10, out_features=len(train_data_simple.classes)
).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)

# Start time
train_start_time = default_timer()

# Train the model
model_0_result = train(
    model=model_0,
    train_data_loader=train_dataloader_simple,
    test_data_loader=test_dataloader_simple,
    loss_fn=loss_fn,
    optimizer=optimizer,
    accuracy_fn=accuracy_fn,
    device=device,
    epochs=_NUM_EPOCHS,
)

# End time
train_end_time = default_timer()
print(f"Training time: {train_end_time - train_start_time:.2f} seconds")
print(model_0_result)

"""
7.8 Plot the loss curves of model_0
 A loss curve is a plot of the training and test loss values across epochs. 
 It can be used to identify when a model is underfitting (high loss), 
 overfitting (training loss much lower than test loss), or just right (training and test loss decrease together).
"""

plot_loss_curves(model_0_result)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/c_line_297_model_0_loss_curves.png"
)

"""
7.9 Save model_0 weights and training results for use in comparison files
"""

# Directory to persist artefacts
_SAVE_DIR = Path("lessons/section6_pytorch_custom_datasets/src")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Save model state dict (weights only — portable across scripts)
_MODEL_PATH = _SAVE_DIR / "c_model_0.pth"
torch.save(model_0.state_dict(), _MODEL_PATH)
print(f"Model saved to: {_MODEL_PATH}")

# Save training results as JSON so any script can load and compare metrics
_RESULTS_PATH = _SAVE_DIR / "c_model_0_results.json"
with open(_RESULTS_PATH, "w") as f:
    json.dump(model_0_result, f, indent=2)
print(f"Results saved to: {_RESULTS_PATH}")

"""
8. What should be ideal loss curve look like?
- Both training and test loss should decrease over time.
- If training loss decreases but test loss plateaus or increases, it may indicate overfitting.
- If both training and test loss are high and don't decrease, it may indicate underfitting.
- The ideal loss curve is one where both training and test loss decrease and eventually plateau at a low value, indicating good generalization to unseen data.  
"""
