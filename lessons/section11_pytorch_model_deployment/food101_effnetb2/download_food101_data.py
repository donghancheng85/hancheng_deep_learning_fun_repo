"""
Download the Food-101 dataset and print a data summary.

Dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
  - 101 food categories
  - 75,750 training images  (750 per class)
  - 25,250 test images      (250 per class)

Data will be stored in: lessons/section11_pytorch_model_deployment/data/

NOTE: torchvision's hardcoded MD5 for Food-101 is stale (the ETH Zurich server
updated the archive). We download manually with wget (resume-safe) and then load
the dataset with download=False to skip the broken MD5 check.
"""

import math
import subprocess
import tarfile
from pathlib import Path

import torchvision

# ── Config ────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("lessons/section11_pytorch_model_deployment/data")
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
TAR_PATH = DATA_DIR / "food-101.tar.gz"
EXTRACTED_DIR = DATA_DIR / "food-101"
BATCH_SIZE = 32

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Download (skip if already extracted) ─────────────────────────────────────────────
if EXTRACTED_DIR.exists():
    print(f"[INFO] food-101/ already extracted at {EXTRACTED_DIR} — skipping download.")
else:
    print(f"[INFO] Downloading Food-101 → {TAR_PATH}")
    print("       (~5 GB; wget resumes automatically if interrupted)\n")
    # -c = resume, --show-progress = progress bar
    subprocess.run(
        ["wget", "-c", "--show-progress", "-O", str(TAR_PATH), FOOD101_URL],
        check=True,
    )
    print(f"\n[INFO] Extracting {TAR_PATH} …")
    with tarfile.open(TAR_PATH) as tar:
        tar.extractall(path=DATA_DIR)
    print(f"[INFO] Extracted → {EXTRACTED_DIR}")
    TAR_PATH.unlink()  # remove archive to save ~5 GB disk space
    print(f"[INFO] Removed archive {TAR_PATH}")

# ── Load datasets (download=False — archive already extracted above) ──────────────────
print("\n[INFO] Loading dataset metadata …")
train_data = torchvision.datasets.Food101(
    root=str(DATA_DIR),
    split="train",
    download=False,
)
test_data = torchvision.datasets.Food101(
    root=str(DATA_DIR),
    split="test",
    download=False,
)

# ── Summary ───────────────────────────────────────────────────────────────────────────
n_train = len(train_data)
n_test = len(test_data)
n_classes = len(train_data.classes)
n_train_batches = math.ceil(n_train / BATCH_SIZE)
n_test_batches = math.ceil(n_test / BATCH_SIZE)

print("\n── Food-101 Dataset Summary ──────────────────────────────")
print(f"  Classes              : {n_classes}")
print(f"  Training images      : {n_train:,}")
print(f"  Test images          : {n_test:,}")
print(f"  Total images         : {n_train + n_test:,}")
print(f"\n  Batch size           : {BATCH_SIZE}")
print(
    f"  Training batches     : {n_train_batches:,}  "
    f"(last batch has {n_train % BATCH_SIZE or BATCH_SIZE} images)"
)
print(
    f"  Test batches         : {n_test_batches:,}  "
    f"(last batch has {n_test % BATCH_SIZE or BATCH_SIZE} images)"
)
print(f"\n  Data stored at       : {DATA_DIR.resolve()}")
print("──────────────────────────────────────────────────────────")
