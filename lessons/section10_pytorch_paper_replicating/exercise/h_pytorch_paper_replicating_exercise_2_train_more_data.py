import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Hyperparameters ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3  # pizza / steak / sushi
EPOCHS = 10
LEARNING_RATE = 1e-3

# ── Data ──────────────────────────────────────────────────────────────────────────────
data_path = Path("lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased")
train_dir = data_path / "train"
test_dir = data_path / "test"

# ── 1. Load pretrained ViT-B/16 from torchvision ─────────────────────────────────────
# ViT_B_16_Weights.IMAGENET1K_V1 — pretrained on ImageNet-1k (1000 classes)
weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
model = torchvision.models.vit_b_16(weights=weights)

# Use the transform defined by the weights — guaranteed to match what the model
# was pretrained with (resize, crop, normalisation values etc.).
# This is safer than a manual v2.Compose: if you swap weights, the transform
# automatically updates to match the new checkpoint.
pretrained_transform = weights.transforms()
print(f"\nPretrained transform: {pretrained_transform}")

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=pretrained_transform,
    test_transform=pretrained_transform,
    batch_size=BATCH_SIZE,
)
print(f"Class names: {class_names}")
print(f"Train batches: {len(train_dataloader)} | Test batches: {len(test_dataloader)}")

# ── 1a. Print original model structure with torchinfo ────────────────────────────────
print("\n── Original pretrained ViT-B/16 structure ──")
summary(
    model=model,
    input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# ── 2. Freeze all parameters (backbone stays frozen) ─────────────────────────────────
for param in model.parameters():
    param.requires_grad = False

# ── 2a. Replace the classifier head for 3-class FoodVision Mini ──────────────────────
# The original head is model.heads.head: Linear(768 → 1000)
# We replace it with a new Linear(768 → 3) — requires_grad=True by default
in_features = model.heads.head.in_features  # 768 for ViT-B/16
model.heads.head = nn.Linear(in_features=in_features, out_features=NUM_CLASSES)
# New layer is trainable; all other parameters remain frozen

model = model.to(device)

print("\n── Updated ViT-B/16 with 3-class head ──")
summary(
    model=model,
    input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# Verify trainable vs frozen parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"\nTotal parameters  : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}  (classifier head only)")
print(f"Frozen params     : {frozen_params:,}  (pretrained backbone)")

# ── 3. Loss, Optimizer, Scheduler ────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss()

# Only pass the trainable classifier parameters to the optimizer
optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

# ── 3a. TensorBoard writer ────────────────────────────────────────────────────────────
log_dir = (
    Path("lessons/section10_pytorch_paper_replicating/runs")
    / f"ViT_pretrained_doubled_data_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
)
writer = SummaryWriter(log_dir=str(log_dir))
print(f"\nTensorBoard logs → {log_dir}")

# ── 3b. Train ─────────────────────────────────────────────────────────────────────────
set_seeds()
print(f"\nFine-tuning ViT-B/16 classifier for {EPOCHS} epochs on {device}...")
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

# ── 4. Plot & Save ────────────────────────────────────────────────────────────────────
plot_loss_curves(results)

save_dir = Path("lessons/section10_pytorch_paper_replicating/models")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = (
    save_dir
    / f"vit_b16_pretrained_pizza_steak_sushi_data_doubled_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
)
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")
