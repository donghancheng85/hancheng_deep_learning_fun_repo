import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from torch import nn

from torchinfo import summary

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# set up device
device = get_best_device()
print_device_info(device)

# Setup path to data folder
data_path = Path("going_modular/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

print(f"torchvision version: {torchvision.__version__}")

"""
1. We want to use ViT (Vision Transformer) for image classification. Using the FoodVision mini
pizza, steak, sushi
"""

"""
2. Create datasets and DataLoaders
"""
# Create Image Size
IMAGE_SIZE = 224
BATCH_SIZE = 32

manual_transform = v2.Compose(
    [
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=manual_transform,
    test_transform=manual_transform,
    batch_size=BATCH_SIZE,
)

print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")
print(f"Number of batches in train dataloader: {len(train_dataloader)}")
print(f"Number of batches in test dataloader: {len(test_dataloader)}")

"""
3. Visualize a single image from train_dataloader
"""
images, labels = next(iter(train_dataloader))
image = images[0]  # shape: [C, H, W], float32 in [0, 1]

fig, ax = plt.subplots()
ax.imshow(image.permute(1, 2, 0))  # CHW -> HWC
ax.set_title(f"Label: {class_names[labels[0]]}")
ax.axis("off")

save_path = Path(
    "lessons/section10_pytorch_paper_replicating/src/a_line_72_train_sample_image.png"
)
fig.savefig(save_path)
plt.close(fig)
print(f"Saved sample image to {save_path}")
