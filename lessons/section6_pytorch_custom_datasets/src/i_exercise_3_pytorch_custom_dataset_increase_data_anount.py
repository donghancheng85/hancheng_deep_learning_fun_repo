import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGG,
)

from timeit import default_timer
import requests
import zipfile
from pathlib import Path
import os
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

# Create train transforms
train_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
    ]
)

test_transform = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.ToImage(),  # PIL Image → uint8 tensor [C, H, W]
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] → float32 [0.0, 1.0]
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
NUM_WORKERS = min(os.cpu_count() or 1, 4)
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

# Create model and train for 20 epochs
model_1_more_data = TinyVGG(
    in_features=3,
    hidden_units=64,
    out_features=len(train_dataset.classes),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1_more_data.parameters(), lr=0.001)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 50 will make mode overfit (train accuracy near 100%, test between 50-60%),
# 20 is a good number to see the difference between more data and less data
NUM_EPOCHS = 20
train_time_start_on_augmented = default_timer()
model_1_more_data_results = train(
    model=model_1_more_data,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    epochs=NUM_EPOCHS,
    device=device,
)
train_time_end_on_augmented = default_timer()
print_train_time(train_time_start_on_augmented, train_time_end_on_augmented)
plot_loss_curves(model_1_more_data_results)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/i_line_136_loss_curves_more_data.png"
)
