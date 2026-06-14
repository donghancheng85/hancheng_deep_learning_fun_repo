import torch
from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time

from lessons.section11_pytorch_model_deployment.model_training.vit_b_16_model_creater import (
    create_vit_b16_model,
)

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Hyperparameters ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224  # ViT-B/16 native input size
BATCH_SIZE = 32
NUM_CLASSES = 3  # pizza / steak / sushi
EPOCHS = 10
LEARNING_RATE = 1e-3

# ── Data ──────────────────────────────────────────────────────────────────────────────
data_path = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)
train_dir = data_path / "train"
test_dir = data_path / "test"

# ── Model ─────────────────────────────────────────────────────────────────────────────
model, pretrained_transform = create_vit_b16_model(num_classes=NUM_CLASSES, seed=42)
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

# ── Model structure ───────────────────────────────────────────────────────────────────
print("\n── ViT-B/16 with 3-class head ──")
summary(
    model=model,
    input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"\nTotal parameters  : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}  (classifier head only)")
print(f"Frozen params     : {frozen_params:,}  (pretrained backbone)")

model = model.to(device)

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss()

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
    "lessons/section11_pytorch_model_deployment/runs/vit_b16"
) / time.strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(log_dir=str(log_dir))
print(f"\nTensorBoard logs → {log_dir}")

# ── Train ─────────────────────────────────────────────────────────────────────────────
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

# ── Plot ──────────────────────────────────────────────────────────────────────────────
plot_loss_curves(results)

# ── Save model ────────────────────────────────────────────────────────────────────────
save_dir = Path("lessons/section11_pytorch_model_deployment/models/vit_b16")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = (
    save_dir / f"vit_b16_pizza_steak_sushi_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
)
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")

# ── Final statistics ──────────────────────────────────────────────────────────────────
final_train_loss = results["train_loss"][-1]
final_train_acc = results["train_accuracy"][-1]
final_test_loss = results["test_loss"][-1]
final_test_acc = results["test_accuracy"][-1]
model_size_mb = save_path.stat().st_size / (1024 * 1024)

print("\n── Final Model Statistics ──")
print(f"  Train loss      : {final_train_loss:.5f}")
print(f"  Train accuracy  : {final_train_acc:.2f}%")
print(f"  Test  loss      : {final_test_loss:.5f}")
print(f"  Test  accuracy  : {final_test_acc:.2f}%")
print(f"  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Model size      : {model_size_mb:.2f} MB")
