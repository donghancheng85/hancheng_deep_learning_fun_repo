from typing import cast
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import v2
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
)

from pathlib import Path
import os
from timeit import default_timer
import matplotlib.pyplot as plt

"""
Why TinyVGG (trained from scratch) struggles on this dataset
─────────────────────────────────────────────────────────────
1. Model capacity is too low (hidden_units=10).
2. No BatchNorm / Dropout → noisy, unstable test curves.
3. No LR scheduler → constant LR causes the oscillating test loss spikes.
4. 64×64 input loses fine-grained texture/colour features.
5. Small dataset (~75 images/class) + random init = not enough signal to
   learn good features from scratch.

Solution: Transfer Learning with EfficientNet-B0
─────────────────────────────────────────────────
• Load EfficientNet-B0 with ImageNet-pretrained weights.
• Freeze the entire feature backbone (no gradients flow through it).
• Replace only the final classifier head with a new Linear layer sized
  to our number of classes (3: pizza / steak / sushi).
• Use the transforms recommended by the pretrained weights (224×224,
  ImageNet mean/std normalisation) so the backbone sees inputs in the
  same distribution it was trained on.
• A cosine-annealing LR scheduler smoothly decays LR, avoiding the
  oscillation seen with a fixed rate.

Result: the frozen backbone acts as a powerful fixed feature extractor
(trained on 1.2M images), so even 20 epochs on our tiny dataset is
enough to reach 85–95 % test accuracy.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Device
# ─────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data paths  (assumes data already downloaded by a_135_to_x_*.py)
# ─────────────────────────────────────────────────────────────────────────────
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
train_dir = data_path / "pizza_steak_sushi" / "train"
test_dir = data_path / "pizza_steak_sushi" / "test"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Transforms
#    Use the auto-transforms bundled with EfficientNet_B0_Weights so we get
#    exactly the same preprocessing that was used during ImageNet training.
#    For training we add a few augmentations on top.
# ─────────────────────────────────────────────────────────────────────────────
weights = EfficientNet_B0_Weights.DEFAULT  # best available pretrained weights
auto_transforms = weights.transforms()  # Resize(256) → CenterCrop(224) + Normalize

# Build train transform: auto_transforms already include Normalize,
# so we add geometric augmentations *before* the auto pipeline.
train_transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(-15, 15)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        auto_transforms,
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Datasets & DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=auto_transforms)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() or 1

torch.manual_seed(42)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build the transfer-learning model
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)

# 4a. Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights=weights).to(device)

# 4b. Freeze ALL backbone parameters so we only train the head.
#     This is critical when the dataset is small — updating the backbone
#     with so few samples would destroy the pretrained features (catastrophic
#     forgetting).
for param in model.features.parameters():
    param.requires_grad = False

# 4c. Replace the classifier head.
#     EfficientNet-B0's original head outputs 1000 classes (ImageNet).
#     model.classifier is:
#       Sequential(Dropout(p=0.2), Linear(1280, 1000))
#     We swap it for a head with our number of classes.
assert isinstance(model.classifier[1], nn.Linear)
in_features = cast(nn.Linear, model.classifier[1]).in_features  # 1280
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=in_features, out_features=len(class_names)),
).to(device)

print(model)
torchinfo.summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Loss, optimiser, and LR scheduler
#    • Only pass the classifier parameters to the optimiser — the backbone
#      weights are frozen so there is no point computing gradients for them.
#    • CosineAnnealingLR decays LR smoothly, avoiding the oscillating test
#      loss spikes seen with a fixed LR in the TinyVGG experiment.
# ─────────────────────────────────────────────────────────────────────────────
NUM_EPOCHS = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.classifier.parameters(),  # only train the new head
    lr=0.001,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_time = default_timer()
results = train(
    model=model,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    accuracy_fn=accuracy_fn,
    epochs=NUM_EPOCHS,
    scheduler=scheduler,
)
end_time = default_timer()
print_train_time(start_time, end_time, device=device)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot & save
# ─────────────────────────────────────────────────────────────────────────────
print(results)
plot_loss_curves(results)
plt.tight_layout()
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/e_line_162_efficientnet_loss_curves.png"
)
