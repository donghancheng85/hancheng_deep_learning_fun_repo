import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pathlib import Path

from common.device import get_best_device, print_device_info

# SVHN (Street View House Numbers): 32x32 RGB, 10 digit classes
# Downloads from Stanford/Google CDN — much faster than UofT CIFAR-10 server

DATA_DIR = Path("lessons/section12_pytorch20_intro/data")
IMAGE_DIR = Path("lessons/section12_pytorch20_intro/images")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.ToTensor()

train_data = datasets.SVHN(
    root=DATA_DIR, split="train", download=True, transform=transform
)
test_data = datasets.SVHN(
    root=DATA_DIR, split="test", download=True, transform=transform
)

SVHN_CLASSES = [str(i) for i in range(10)]  # digits 0–9

# --- Print statistics ---
print("\n--- SVHN Dataset Statistics ---")
print(f"Classes ({len(SVHN_CLASSES)}): {SVHN_CLASSES}")
print(f"Train samples : {len(train_data)}")
print(f"Test  samples : {len(test_data)}")

sample_img, sample_label = train_data[0]
print(f"Image shape   : {sample_img.shape}  (C x H x W)")
print(f"Image dtype   : {sample_img.dtype}")
print(f"Pixel range   : [{sample_img.min():.4f}, {sample_img.max():.4f}]")
print(f"Sample label  : {sample_label} -> {SVHN_CLASSES[sample_label]}")

# --- Plot one sample per class (10 images) ---
# Find the first occurrence of each digit in the training set
class_to_idx: dict[int, int] = {}
for idx, (_, label) in enumerate(train_data):
    if label not in class_to_idx:
        class_to_idx[label] = idx
    if len(class_to_idx) == 10:
        break

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("SVHN – one sample per digit class", fontsize=14)

for digit in range(10):
    img, label = train_data[class_to_idx[digit]]
    img_np = img.permute(1, 2, 0).numpy()  # C×H×W → H×W×C for matplotlib
    ax = axes[digit // 5][digit % 5]
    ax.imshow(img_np)
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
save_path = IMAGE_DIR / "svhn_sample_per_class.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nSample image saved → {save_path}")
