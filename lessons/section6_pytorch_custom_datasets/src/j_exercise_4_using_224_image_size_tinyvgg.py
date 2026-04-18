import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGGWithCustomImageShape,
)

import requests
import zipfile
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt

"""
7. Double the data you're using with your model and train it for 20 epochs, what happens to the results?
Note: You can use the custom data creation notebook to scale up your Food101 dataset.
You can also find the already formatted double data (20% instead of 10% subset) dataset on GitHub, you will need to write download code like in exercise 2 to get it into this notebook.
"""

# get best device for training
device = get_best_device()
print_device_info(device)

# Set up path to a data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi_increased"
zip_path = data_path / "pizza_steak_sushi_increased.zip"

# Download zip if it doesn't exist yet
if zip_path.exists():
    print(f"{zip_path} already exists... skipping download")
else:
    data_path.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        response = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi_20_percent.zip"
        )
        print("Downloading data...")
        f.write(response.content)

# Unzip if the extracted folder doesn't exist yet
if image_path.is_dir():
    print(f"{image_path} already exists... skipping unzip")
else:
    image_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping data...")
        zip_ref.extractall(image_path)  # Unzip the downloaded file

# Train and test directories
train_dir = image_path / "train"
test_dir = image_path / "test"

# ImageNet mean/std — improves gradient flow and generalization for natural images
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Create train transforms
# Extra augmentations act as implicit regularization to reduce overfitting
train_transform = v2.Compose(
    [
        v2.Resize(size=(224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),  # small rotations
        v2.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),  # colour variations
        v2.RandomGrayscale(p=0.1),  # occasional grayscale
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.RandomErasing(p=0.2),  # randomly mask patches to reduce overfitting
    ]
)

test_transform = v2.Compose(
    [
        v2.Resize(size=(224, 224)),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Use ImageFolder to create datasets
train_dataset = datasets.ImageFolder(
    root=train_dir, transform=train_transform, target_transform=None
)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Create DataLoaders
BATCH_SIZE = 32
# Cap workers to avoid exhausting /dev/shm on Linux, which causes a segfault.
# os.cpu_count() can be large; 2-4 workers is typically safe for image datasets.
NUM_WORKERS = min(os.cpu_count() or 1, 2)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")


# plot 9 random images with 224x224 image size, the new image size we're using for this model
def plot_random_images_from_dataloader(
    dataloader: DataLoader, class_names: list[str]
) -> None:
    """Plots 9 random images from a dataloader."""
    # Get a batch of data
    images, labels = next(iter(dataloader))
    # Select 9 random indices
    random_indices = random.sample(range(len(images)), 9)
    # Plot the images
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(random_indices):
        ax = plt.subplot(3, 3, i + 1)
        img = images[idx].permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
        label = class_names[labels[idx]]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")


# plot_random_images_from_dataloader(train_dataloader, train_dataset.classes)
# plt.savefig(
#     "lessons/section6_pytorch_custom_datasets/src/j_line_130_random_images_from_dataloader_higher_resolution.png"
# )

# Create model and train for 20 epochs
model_1_more_data_higher_resolution = TinyVGGWithCustomImageShape(
    in_features=3,
    hidden_units=64,
    out_features=len(train_dataset.classes),
    image_height=224,
    image_width=224,
).to(device)

# Print info about our model
torchinfo.summary(model_1_more_data_higher_resolution, input_size=(32, 3, 224, 224))

loss_fn = nn.CrossEntropyLoss()
# weight_decay adds L2 regularisation — penalises large weights without touching the model
optimizer = torch.optim.Adam(
    model_1_more_data_higher_resolution.parameters(), lr=0.001, weight_decay=1e-4
)

NUM_EPOCHS = 50
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# CosineAnnealingLR decays lr smoothly from 0.001 to near-0 over training,
# preventing the optimizer from overfitting late epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

model_1_more_data_higher_resolution_results = train(
    model=model_1_more_data_higher_resolution,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    epochs=NUM_EPOCHS,
    device=device,
    scheduler=scheduler,
)
plot_loss_curves(model_1_more_data_higher_resolution_results)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/j_line_188_loss_curves_more_data_higher_resolution.png"
)

# Save model state dict (weights only — portable across scripts)
# To reload: instantiate TinyVGGWithCustomImageShape with the same args, then call
#   model.load_state_dict(torch.load(_MODEL_PATH, map_location=device))
_SAVE_DIR = Path("lessons/section6_pytorch_custom_datasets/src")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)
_MODEL_PATH = _SAVE_DIR / "j_model_tinyvgg_224.pth"
torch.save(model_1_more_data_higher_resolution.state_dict(), _MODEL_PATH)
print(f"Model saved to: {_MODEL_PATH}")
