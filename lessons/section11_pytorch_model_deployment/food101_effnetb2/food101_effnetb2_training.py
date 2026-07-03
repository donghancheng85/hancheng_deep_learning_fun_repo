"""
Fine-tune EfficientNet-B2 on Food-101 (101 classes).

Key design decisions:
  - Data augmentation: composed on top of the pretrained normalisation values
    so the backbone always sees correctly-normalised pixels.
  - Label smoothing (ε = 0.1): reduces over-confidence, helps generalisation
    on large datasets like Food-101.
  - TensorBoard: logs loss & accuracy curves to
    lessons/section11_pytorch_model_deployment/runs/food101_effnetb2/<timestamp>
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from lessons.section11_pytorch_model_deployment.model_training.effnet_b2_model_creater import (
    create_effnet_b2_model,
)
from going_modular.pytorch_project import engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Hyperparameters ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 288  # EfficientNet-B2 native input size
BATCH_SIZE = 32
NUM_CLASSES = 101  # Food-101 has 101 categories
EPOCHS = 10
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1  # ε — fraction of probability mass distributed to wrong classes
NUM_WORKERS = 4

# ── Data paths ────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("lessons/section11_pytorch_model_deployment/data")

# ── Model + pretrained transform ──────────────────────────────────────────────────────
# Create first so we can read the normalisation parameters from the weights.
model, pretrained_transform = create_effnet_b2_model(num_classes=NUM_CLASSES, seed=42)

# ImageNet normalisation constants — fixed for EfficientNet_B2_Weights.IMAGENET1K_V1.
# Hardcoded to avoid introspecting the old-style transforms.Compose return type.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ───────────────────────────────────────────────────────────────────────
# Training: augmentation applied BEFORE normalisation.
# Random resized crop ≈ random zoom + random crop; H-flip + colour jitter add variety.
train_transform = v2.Compose(
    [
        v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.RandomRotation(degrees=(-15, 15)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Test: deterministic resize + centre-crop (same as pretrained_transform but explicit).
test_transform = v2.Compose(
    [
        v2.Resize(IMAGE_SIZE + 16),  # slight over-scale then centre-crop
        v2.CenterCrop(IMAGE_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

print(f"\nTrain transform:\n{train_transform}")
print(f"\nTest transform:\n{test_transform}")

# ── Datasets ──────────────────────────────────────────────────────────────────────────
# Food101 splits into train/test using the official meta/train.txt & meta/test.txt.
# We pass transform=None and apply our own via the transform argument so Pillow images
# are produced first (required by v2 transforms).
train_dataset = datasets.Food101(
    root=str(DATA_DIR),
    split="train",
    transform=train_transform,
    download=False,
)
test_dataset = datasets.Food101(
    root=str(DATA_DIR),
    split="test",
    transform=test_transform,
    download=False,
)

class_names = train_dataset.classes
print(f"\nClasses: {len(class_names)} | e.g. {class_names[:5]} … {class_names[-3:]}")

# ── DataLoaders ───────────────────────────────────────────────────────────────────────
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

print(f"Train batches : {len(train_dataloader):,}  ({len(train_dataset):,} images)")
print(f"Test  batches : {len(test_dataloader):,}  ({len(test_dataset):,} images)")

# ── Model structure ───────────────────────────────────────────────────────────────────
print("\n── EfficientNet-B2 — 101-class head ──")
summary(
    model=model,
    input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters  : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}  (classifier head only)")
print(f"Frozen params     : {total_params - trainable_params:,}  (pretrained backbone)")

model = model.to(device)

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────────────
# label_smoothing=0.1 means 10 % of the probability mass is spread uniformly
# across all wrong classes, preventing over-confident predictions.
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

# ── TensorBoard writer ────────────────────────────────────────────────────────────────
log_dir = Path(
    "lessons/section11_pytorch_model_deployment/runs/food101_effnetb2"
) / time.strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(log_dir=str(log_dir))
print(f"\nTensorBoard logs → {log_dir}")

# ── Train ─────────────────────────────────────────────────────────────────────────────
set_seeds()
print(f"\nFine-tuning EfficientNet-B2 on Food-101 for {EPOCHS} epochs on {device}…")
train_start = time.perf_counter()

results = engine.train_for_summarywriter(
    model=model,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    device=device,
    accuracy_fn=accuracy_fn,
    writer=writer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    scheduler=scheduler,
)

train_end = time.perf_counter()
total_seconds = train_end - train_start
print(
    f"\nTraining time: {total_seconds:.2f}s "
    f"({total_seconds / 60:.2f} min | {total_seconds / EPOCHS:.2f}s/epoch)"
)

# ── Plot ──────────────────────────────────────────────────────────────────────────────
plot_loss_curves(results)

# ── Save model ────────────────────────────────────────────────────────────────────────
save_dir = Path("lessons/section11_pytorch_model_deployment/models/food101_effnetb2")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f"effnet_b2_food101_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
torch.save(model.state_dict(), save_path)
print(f"\nModel saved → {save_path}")

# ── Final statistics ──────────────────────────────────────────────────────────────────
model_size_mb = save_path.stat().st_size / (1024 * 1024)

print("\n── Final Model Statistics ──")
print(f"  Train loss      : {results['train_loss'][-1]:.5f}")
print(f"  Train accuracy  : {results['train_accuracy'][-1]:.2f}%")
print(f"  Test  loss      : {results['test_loss'][-1]:.5f}")
print(f"  Test  accuracy  : {results['test_accuracy'][-1]:.2f}%")
print(f"  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Model size      : {model_size_mb:.2f} MB")
