import torchvision
from pathlib import Path

data_dir = Path("lessons/section9_pytorch_experiment_tracking/data/food101")
data_dir.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Downloading Food101 train split to {data_dir}...")
train_data = torchvision.datasets.Food101(
    root=data_dir,
    split="train",
    download=True,
)

print(f"[INFO] Downloading Food101 test split to {data_dir}...")
test_data = torchvision.datasets.Food101(
    root=data_dir,
    split="test",
    download=True,
)

print(f"[INFO] Train samples: {len(train_data)}")
print(f"[INFO] Test samples : {len(test_data)}")
print(f"[INFO] Classes      : {len(train_data.classes)}")
