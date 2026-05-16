import torch
import torchvision
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from going_modular.pytorch_project import engine, utils
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, create_summary_writer

from lessons.section9_pytorch_experiment_tracking.common.common_functions import (
    create_efficientnet_b3_model,
)

# Set up device
device = get_best_device()
print_device_info(device)

data_dir = Path("lessons/section9_pytorch_experiment_tracking/data/food101")

# EfficientNet B3 default pretrained transform
effnetb3_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()

train_dataset = torchvision.datasets.Food101(
    root=data_dir,
    split="train",
    transform=effnetb3_transform,
    download=False,
)

test_dataset = torchvision.datasets.Food101(
    root=data_dir,
    split="test",
    transform=effnetb3_transform,
    download=False,
)

class_names = train_dataset.classes
print(f"[INFO] Classes: {len(class_names)}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")

BATCH_SIZE = 32
NUM_WORKERS = 8

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Create EfficientNet B3 — features frozen, only classifier head trained
set_seeds(seed=42)
model = create_efficientnet_b3_model(num_classes=len(class_names)).to(device)

EPOCHS = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

writer = create_summary_writer(
    experiment_name="food101",
    model_name=model.__class__.__name__,
    extra=f"{EPOCHS}_epochs",
    index="food101-effnetb3",
)

start_time = timer()
engine.train_for_summarywriter(
    model=model,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    device=device,
    accuracy_fn=accuracy_fn,
    writer=writer,
    scheduler=scheduler,
)
scheduler.step()
end_time = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds ({(end_time - start_time) / 60:.2f} minutes)")

utils.save_model(
    model=model,
    save_path=Path("lessons/section9_pytorch_experiment_tracking/models"),
    model_name="09_efficientnet_b3_food101_10_epochs.pth",
)
