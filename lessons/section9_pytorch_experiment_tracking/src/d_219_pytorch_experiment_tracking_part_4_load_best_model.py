import matplotlib.pyplot as plt
import torch
import torchvision
from pathlib import Path

from torch import nn
from torchvision import transforms
import random


from common.device import get_best_device, print_device_info
from common.helper_fucntion import (
    pred_and_plot_image,
)


# Set up device
device = get_best_device()
print_device_info(device)

# Set up paths
data_path_10_percent = Path("going_modular/data/pizza_steak_sushi")
data_path_20_percent = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)

# Set up training dir
train_dir_10_percent = data_path_10_percent / "train"
train_dir_20_percent = data_path_20_percent / "train"

# Set up test dir (same for both)
test_dir = data_path_10_percent / "test"

# 1. Re-create the model architecture with the same custom classifier head
# (3 output classes: pizza, steak, sushi)
# Use weights=None to skip loading pretrained ImageNet weights — they will be
# fully overwritten by load_state_dict anyway, so loading them is wasteful.
model = torchvision.models.efficientnet_b2(weights=None)
in_features: int = model.classifier[-1].in_features  # type: ignore[union-attr]
model.classifier[-1] = nn.Linear(in_features=in_features, out_features=3)
model = model.to(device)

# 2. Load the saved state_dict into the model
model_path = Path(
    "lessons/section9_pytorch_experiment_tracking/models/09_effnetb2_data_20_percent_10_epochs.pth"
)
model.load_state_dict(torch.load(f=model_path, map_location=device))
print(f"[INFO] Loaded model from: {model_path}")

# 3. Put the model in eval mode for inference
model.eval()
print(f"[INFO] Model is ready for inference: {model.__class__.__name__}")

# 4. Get a random image from test_dir and predict
class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

# Collect all test images across all class subdirectories
all_test_images = list(test_dir.glob("*/*.jpg"))
random_images = random.sample(all_test_images, 9)

# EfficientNet B2 expects 260x260 input
effnetb2_transform = transforms.Compose(
    [
        transforms.Resize((260, 260)),
    ]
)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for ax, image_path in zip(axes.flatten(), random_images):
    true_class = image_path.parent.name
    pred_and_plot_image(
        model=model,
        image_path=image_path,
        class_names=class_names,
        transform=effnetb2_transform,
        device=device,
        ax=ax,
        true_class_name=true_class,
    )
plt.tight_layout()

# 5. Save the figure
save_path = Path(
    "lessons/section9_pytorch_experiment_tracking/src/d_line_96_pred_random_result.png"
)
plt.savefig(save_path, bbox_inches="tight")
print(f"[INFO] Prediction plot saved to: {save_path}")
plt.close(fig)

# Make predictions of a web image
web_pizza_image_path = "lessons/section6_pytorch_custom_datasets/data/04-pizza-dad.jpg"
pred_and_plot_image(
    model=model,
    image_path=web_pizza_image_path,
    class_names=class_names,
    transform=effnetb2_transform,
    device=device,
    ax=None,
    true_class_name="Pizza (web image)",
)

plt.savefig(
    "lessons/section9_pytorch_experiment_tracking/src/d_line_115_web_pizza_prediction.png"
)
