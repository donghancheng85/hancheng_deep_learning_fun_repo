"""
Fine-tune EfficientNet-B2 on Food-101 (101 classes).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGE LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v1 — Head-only transfer learning (2026-07-03)
  • Backbone frozen; only Linear(1408 → 101) head trained
  • Optimizer : Adam, LR=1e-3
  • Scheduler : CosineAnnealingLR, T_max=10
  • Epochs    : 10
  • Result    : Train 52.2% | Test 63.5%  (test > train due to aug)
  • Note      : Label smoothing ε=0.1 suppresses train accuracy metric;
                test > train is expected and healthy here.

v2 — Full fine-tuning with discriminative LR (2026-07-03)
  • All backbone parameters unfrozen
  • Optimizer : Adam, discriminative LR — backbone 1e-4, head 1e-3
  • Scheduler : CosineAnnealingLR, T_max=10
  • Epochs    : 10
  • Result    : Train 94.7% | Test 88.6%  (+25.1 pp on test vs v1)
  • Note      : Overfitting onset at epoch 5 (gap flips positive).

v3 — Stability fix: SafeFood101 wrapper (2026-07-03)
  • Added SafeFood101 dataset wrapper and safe_collate function
  • Catches corrupted JPEGs that caused cudaErrorIllegalAddress crash
  • Reduced NUM_WORKERS 4 → 2
  • No accuracy change (infrastructure fix only)

v4 — AdamW + CosineAnnealingWarmRestarts, 20 epochs (2026-07-04)
  • Optimizer : AdamW, weight_decay=1e-4 (decoupled L2 regularisation)
  • Scheduler : CosineAnnealingWarmRestarts, T_0=10, T_mult=2
  • Epochs    : 20
  • Result    : Train 97.7% | Test 88.5%  (gap: 9.2 pp — MORE overfitting)
  • Note      : Warm restart at epoch 11 re-opened the model and allowed
                further memorisation of training data.  Test accuracy did
                not improve beyond v2.  Bottleneck is overfitting, not the
                optimizer/scheduler.

v5 — MixUp augmentation + increased Dropout (2026-07-04)
  • MixUp alpha=0.2 applied per training batch.
  • Classifier Dropout 0.3 → 0.4.
  • Scheduler reverted to CosineAnnealingLR(T_max=10).
  • Custom training loop for soft-label loss.
  • Result    : Train 84.5% | Test 88.7%  (gap: −3.7 pp — overfitting GONE)
  • Note      : MixUp makes train harder so train < test is expected and healthy.
                Test acc still rising at epoch 10 — not fully converged.
                LR hit near-zero too early (T_max=10 too short).

v6 — Extended training: 20 epochs, T_max=20  (2026-07-04)  ← CURRENT
  • EPOCHS 10 → 20; T_max 10 → 20 so LR decays more slowly over the longer run.
  • No overfitting risk (train < test confirmed in v5) — safe to extend.
  • Expected : ~89.5–90% test accuracy if the ~0.15%/epoch trend continues.
  • Result    : Train 90.31% | Test 89.38%  (best test 89.52% @ epoch 17)
                Training time: 4333.94s (72.23 min | 216.70s/epoch)
                Prediction confirmed: test acc matched the ~89.5–90% target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY DESIGN DECISIONS (current run — v6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Data augmentation: composed on top of the pretrained normalisation
    values so the backbone always sees correctly-normalised pixels.
  - Label smoothing (ε = 0.1): reduces over-confidence, helps
    generalisation on large datasets like Food-101.
  - TensorBoard: logs loss & accuracy curves to
    lessons/section11_pytorch_model_deployment/runs/food101_effnetb2/
"""

import time
from pathlib import Path
from typing import cast

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm.auto import tqdm

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
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularisation via AdamW — decoupled from gradient update
LABEL_SMOOTHING = 0.1  # ε — fraction of probability mass distributed to wrong classes
MIXUP_ALPHA = (
    0.2  # MixUp interpolation strength; 0.2 is standard for image classification
)
DROPOUT_P = (
    0.4  # classifier head dropout (default 0.3 → increased for extra regularisation)
)
NUM_WORKERS = 2  # restored: SafeFood101 tensor validation catches bad files regardless
# of worker count; 0 disabled prefetching and tripled training time

# ── Data paths ────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("lessons/section11_pytorch_model_deployment/data")

# ── Model + pretrained transform ──────────────────────────────────────────────────────
# Create first so we can read the normalisation parameters from the weights.
model, pretrained_transform = create_effnet_b2_model(num_classes=NUM_CLASSES, seed=42)

# ImageNet normalisation constants — fixed for EfficientNet_B2_Weights.IMAGENET1K_V1.
# Hardcoded to avoid introspecting the old-style transforms.Compose return type.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
# ── Safe dataset wrapper ─────────────────────────────────────────────────────────────
# Food-101 contains occasional corrupted JPEGs. A bad file causes a CUDA illegal
# memory access when the malformed tensor is moved to the GPU.
# SafeFood101 catches per-image errors and returns a neighbouring valid sample instead.
class SafeFood101(Dataset):
    """Wraps Food101 and skips corrupted images by returning the next valid sample."""

    def __init__(self, base_dataset: datasets.Food101) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        for offset in range(len(self.base)):
            try:
                sample = self.base[(idx + offset) % len(self.base)]
                img_tensor, label = sample
                # Guard against PIL silently producing malformed tensors from
                # corrupted JPEGs — these cause cudaErrorIllegalAddress on .to(device).
                if not isinstance(img_tensor, torch.Tensor):
                    raise ValueError(f"Expected Tensor, got {type(img_tensor)}")
                if img_tensor.shape != torch.Size([3, IMAGE_SIZE, IMAGE_SIZE]):
                    raise ValueError(f"Unexpected shape {img_tensor.shape}")
                if not torch.isfinite(img_tensor).all():
                    raise ValueError("Tensor contains NaN or Inf")
                return img_tensor, label
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] Skipping corrupted sample idx={(idx + offset) % len(self.base)}: {exc}"
                )
        raise RuntimeError("No valid samples found in dataset")


# ── Collate: drop any None samples that slip through ─────────────────────────────────
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


# are produced first (required by v2 transforms).
train_dataset = SafeFood101(
    datasets.Food101(
        root=str(DATA_DIR),
        split="train",
        transform=train_transform,
        download=False,
    )
)
test_dataset = SafeFood101(
    datasets.Food101(
        root=str(DATA_DIR),
        split="test",
        transform=test_transform,
        download=False,
    )
)

class_names = train_dataset.base.classes
print(f"\nClasses: {len(class_names)} | e.g. {class_names[:5]} … {class_names[-3:]}")

# ── DataLoaders ───────────────────────────────────────────────────────────────────────
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=safe_collate,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=safe_collate,
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
print(f"\nTotal parameters  : {total_params:,}")
print(f"All params trainable after backbone unfreeze: {total_params:,}")

# ── Unfreeze backbone for full fine-tuning ────────────────────────────────────────────
# All 7.8 M parameters become trainable.
# Use discriminative learning rates: backbone gets a 10× lower LR than the head
# to avoid destroying the pretrained features with large gradient updates.
for param in model.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params  : {trainable_params:,}  (full model — backbone + head)")

# Increase classifier dropout: 0.3 (default) → DROPOUT_P for stronger regularisation
_effnet = cast(torchvision.models.EfficientNet, model)
_effnet.classifier[0] = nn.Dropout(p=DROPOUT_P)
print(f"Classifier dropout set to {DROPOUT_P}")

model = model.to(device)

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Discriminative fine-tuning + AdamW weight decay.
_effnet = cast(torchvision.models.EfficientNet, model)
optimizer = torch.optim.AdamW(
    [
        {
            "params": _effnet.features.parameters(),
            "lr": LEARNING_RATE / 10,
            "weight_decay": WEIGHT_DECAY,
        },  # backbone
        {
            "params": _effnet.classifier.parameters(),
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },  # head
    ]
)

# CosineAnnealingLR: T_max matches EPOCHS so LR decays smoothly to zero at the end.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

# MixUp transform — applied per batch inside the training loop.
# alpha=MIXUP_ALPHA controls interpolation strength (0.2 = mild blend).
# v2.MixUp expects integer labels and returns float soft labels.
mixup_fn = v2.MixUp(num_classes=NUM_CLASSES, alpha=MIXUP_ALPHA)

# ── TensorBoard writer ────────────────────────────────────────────────────────────────
log_dir = Path(
    "lessons/section11_pytorch_model_deployment/runs/food101_effnetb2_mixup_20ep"
) / time.strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(log_dir=str(log_dir))
print(f"\nTensorBoard logs → {log_dir}")

# ── Train ─────────────────────────────────────────────────────────────────────────────
# Custom loop to handle MixUp soft labels correctly:
#   - Loss   : uses float soft labels from MixUp (CE supports this since PyTorch 1.10+)
#   - Accuracy: uses argmax(soft_labels) as the dominant class — approximate but standard
#   - Test   : no MixUp; uses engine.test_step as usual (exact accuracy)
set_seeds()
print(f"\nFine-tuning EfficientNet-B2 on Food-101 for {EPOCHS} epochs on {device}…")
train_start = time.perf_counter()

results: dict[str, list[float]] = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    # ── Training step with MixUp ──────────────────────────────────────────────────────
    model.train()
    epoch_train_loss, epoch_train_acc = 0.0, 0.0

    print(f"Epoch {epoch + 1}/{EPOCHS}", flush=True)
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        X_mix, y_soft = mixup_fn(X, y)  # y_soft: (B, 101) float soft labels

        optimizer.zero_grad()
        logits = model(X_mix)
        loss = loss_fn(logits, y_soft)  # CE accepts float labels directly
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        # Accuracy: dominant mixed class vs predicted class (approximate metric)
        epoch_train_acc += accuracy_fn(y_soft.argmax(dim=1), logits.argmax(dim=1))

    epoch_train_loss /= len(train_dataloader)
    epoch_train_acc /= len(train_dataloader)
    tqdm.write(
        f"Train loss: {epoch_train_loss:.5f} | Train accuracy: {epoch_train_acc:.2f}%"
    )

    # ── Test step (no MixUp) ─────────────────────────────────────────────────────────
    epoch_test_loss, epoch_test_acc = engine.test_step(
        model=model,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )

    scheduler.step()

    # ── TensorBoard ──────────────────────────────────────────────────────────────────
    writer.add_scalars(
        "Loss",
        {"train_loss": epoch_train_loss, "test_loss": epoch_test_loss},
        global_step=epoch,
    )
    writer.add_scalars(
        "Accuracy",
        {"train_accuracy": epoch_train_acc, "test_accuracy": epoch_test_acc},
        global_step=epoch,
    )

    results["train_loss"].append(epoch_train_loss)
    results["train_accuracy"].append(epoch_train_acc)
    results["test_loss"].append(epoch_test_loss)
    results["test_accuracy"].append(epoch_test_acc)

    tqdm.write(
        f"Epoch {epoch + 1} metrics:\n"
        f"Train loss: {epoch_train_loss:.5f}, Train accuracy: {epoch_train_acc:.4f}%\n"
        f"Test loss: {epoch_test_loss:.5f}, Test accuracy: {epoch_test_acc:.4f}%\n"
        + "-"
        * 32
    )

writer.close()

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
save_path = (
    save_dir / f"effnet_b2_food101_mixup_20ep_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
)
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
