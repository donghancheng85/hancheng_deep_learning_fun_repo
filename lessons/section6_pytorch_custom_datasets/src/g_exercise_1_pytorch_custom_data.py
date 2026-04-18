import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGG,
)

from timeit import default_timer
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt

"""
1. Our models are underperforming (not fitting the data well). 
   What are 3 methods for preventing underfitting? Write them down and explain each with a sentence.

- Increase model complexity: Use a more complex model with more parameters to better capture the underlying patterns in the data.
- Feature engineering: Add more relevant features or transform existing features to provide the model with more informative inputs.
- Reduce regularization: Decrease the strength of regularization techniques like dropout or weight decay to allow the model to fit the training data more closely.
"""


"""
2. Recreate the data loading functions we built in sections 1, 2, 3 and 4. You should have train and test DataLoader's ready to use.
"""
# get best device for training
device = get_best_device()
print_device_info(device)

# Set up path to a data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"
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

# Visualize a sample from the training dataset,using random number generator to get a random index
random_index = random.randint(0, len(train_dataset) - 1)
class_names = train_dataset.classes
print(f"Class names: {class_names}")
image, label = train_dataset[random_index]
print(
    f"Image shape: {image.shape} | Label: {label} | Class name: {class_names[label]} | data type: {image.dtype} | label data type: {type(label)}"
)
# Permute the image from [C, H, W] to [H, W, C] for visualization
image_permute = image.permute(1, 2, 0)
print(
    f"Original image shape: {image.shape} -> [C, H, W] | Permuted image shape: {image_permute.shape} -> [H, W, C]"
)
plt.figure(figsize=(6, 6))
plt.imshow(image_permute)
plt.axis(False)
plt.title(f"Class: {class_names[label]} | Shape: {image.shape} -> [C, H, W]")
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/g_line_88_visualize_image_from_imagefolder.png"
)

"""
3. Recreate model_0 we built in section 7.
"""

model_0 = TinyVGG(
    in_features=3,
    hidden_units=10,
    out_features=len(train_dataset.classes),
)

# inspect the model architecture
torchinfo.summary(model_0, input_size=(1, 3, 64, 64))


"""
4. Create training and testing functions for model_0.
(done by importing train and plot_loss_curves from common.py)
"""

"""
5. Try training the model you made in exercise 3 for 5, 20 and 50 epochs, what happens to the results?
   Use torch.optim.Adam() with a learning rate of 0.001 as the optimizer.
"""
loss_fn = nn.CrossEntropyLoss()

NUM_EPOCHS: list = [5, 20, 50]

train_results: list[dict[str, list[float] | str]] = []
train_times: list[float] = []

for epoch in NUM_EPOCHS:
    # Recreate model and optimizer from scratch each run so the three
    # experiments (5 / 20 / 50 epochs) are fully independent.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model_0 = TinyVGG(
        in_features=3,
        hidden_units=10,
        out_features=len(train_dataset.classes),
    ).to(device)
    optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)
    start_time = default_timer()
    train_results.append(
        train(
            model=model_0,
            train_data_loader=train_dataloader,
            test_data_loader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            accuracy_fn=accuracy_fn,
            epochs=epoch,
        )
    )
    end_time = default_timer()
    train_times.append(print_train_time(start_time, end_time, device=device))
    print(f"Training time for {epoch} epochs: {train_times[-1]}\n\n")

# Plot the loss curves for each epoch setting
for i, epoch in enumerate(NUM_EPOCHS):
    print(
        f"Results for {epoch} epochs: {train_results[i]} | training time: {train_times[i]}"
    )
    plot_loss_curves(train_results[i])
    plt.tight_layout()
    plt.savefig(
        f"lessons/section6_pytorch_custom_datasets/src/g_line_162_model_0_loss_curves_{epoch}_epochs.png"
    )
