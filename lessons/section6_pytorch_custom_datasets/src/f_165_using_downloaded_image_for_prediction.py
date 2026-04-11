import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import torchvision
import torchinfo

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, print_train_time
from lessons.section6_pytorch_custom_datasets.common.common import (
    train,
    plot_loss_curves,
    TinyVGG,
)

from typing import Tuple, Dict, List
from pathlib import Path
import pathlib
import os
import random
from PIL import Image
from timeit import default_timer
import matplotlib.pyplot as plt
import requests

device = get_best_device()
print_device_info(device)
"""
Load the model weights from d_161_pytorch_custom_dataset_model_with_augment.py 
and use it to make a prediction on a single image downloaded from the internet.
"""
model_1 = TinyVGG(in_features=3, hidden_units=10, out_features=3).to(device)
_MODEL_PATH = Path("lessons/section6_pytorch_custom_datasets/src/d_model_1.pth")
model_1.load_state_dict(torch.load(_MODEL_PATH, map_location=device, weights_only=True))
model_1.eval()

"""
11. Making a prediction on a custom image downloaded from the internet

Image is not in either the train or test set, but we can still make a prediction on it using our trained model.
"""

# Download an image from the internet and save it to disk (done manually in this case)
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

# Load the train image dataset to get the class names (pizza, steak, sushi)
train_transform_trivial = v2.Compose(
    [
        v2.Resize((64, 64)),
        v2.TrivialAugmentWide(num_magnitude_bins=31),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
train_dataset_augmented = datasets.ImageFolder(
    root=train_dir, transform=train_transform_trivial
)

custom_image_path = data_path / "04-pizza-dad.jpg"

if not custom_image_path.exists():
    with open(custom_image_path, "wb") as f:
        f.write(
            requests.get(
                "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg"
            ).content
        )
        print(f"Custom image downloaded and saved to: {custom_image_path}")
else:
    print(f"Custom image already exists at: {custom_image_path}")


"""
11.1 Loading the custom image and making a prediction on it

- In tensor form with datatype of float32 (scaled between 0 and 1)
- Shape of (1, 3, 64, 64) - a batch of 1 image with 3 color channels and 64x64 pixels
- on target device (GPU or CPU)

Using read_image or decode_image from torchvision.io to load the image, then applying the same transformations as the test set (resize to 64x64, convert to float32 and scale between 0 and 1) before making a prediction with the model.
"""

custom_image_unint8 = torchvision.io.decode_image(str(custom_image_path))
print(
    f"Custom image loaded with shape: {custom_image_unint8.shape} and dtype: {custom_image_unint8.dtype}"
)

"""
11.2 Making a prediction on the custom image with trained PyTorch model
"""
# Try to make prediction on a image in uint8 format (values between 0 and 255, this will be incorrect)
model_1.eval()
try:
    with torch.inference_mode():
        custom_image_unint8 = custom_image_unint8.unsqueeze(0).to(
            device
        )  # Add batch dimension and move to device
        pred_logits = model_1(custom_image_unint8)
        pred_label = pred_logits.argmax(dim=1).item()
        print(f"Predicted label (uint8 image): {pred_label}")
except Exception as e:
    print(f"Error making prediction on uint8 image: {e}")

# Loading the custom image and converting to float32 format (values between 0 and 1) before making a prediction
custom_image_float = torchvision.io.decode_image(str(custom_image_path)).float()
print(
    f"Custom image converted to float with shape: {custom_image_float.shape} and dtype: {custom_image_float.dtype}"
)

model_1.eval()
try:
    with torch.inference_mode():
        custom_image_float = custom_image_float.unsqueeze(0).to(
            device
        )  # Add batch dimension and move to device
        pred_logits = model_1(custom_image_float)
        pred_label = pred_logits.argmax(dim=1).item()
        print(f"Predicted label (float image): {pred_label}")
except Exception as e:
    print(f"Error making prediction on float image: {e}")

# Create transform pipleine for the custom image (resize to 64x64, convert to float32 and scale between 0 and 1)
custom_image_transform = v2.Compose(
    [
        v2.Resize((64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
custom_image = torchvision.io.decode_image(str(custom_image_path))
custom_image_transformed = custom_image_transform(custom_image).to(
    device
)  # Add batch dimension and move to device
print(
    f"Original custom image shape: {custom_image.shape} and dtype: {custom_image.dtype}"
)
print(
    f"Custom image transformed with shape: {custom_image_transformed.shape} and dtype: {custom_image_transformed.dtype}"
)
model_1.eval()
try:
    with torch.inference_mode():
        custom_image_transformed = custom_image_transformed.unsqueeze(0).to(
            device
        )  # Add batch dimension and move to device
        pred_logits = model_1(custom_image_transformed)
        pred_label = pred_logits.argmax(dim=1).item()
        print(f"Predicted label (transformed image): {pred_label}")
        print(f"Predicted label name: {train_dataset_augmented.classes[pred_label]}")
except Exception as e:
    print(f"Error making prediction on transformed image: {e}")

"""
11.3 Building a function for custom image prediction

A function pass an image path and a trained model, then load the image, apply the necessary transformations and make a prediction with the model. 
The function should return the predicted class label name.
"""


def predict_custom_image_and_plot(
    image_path: Path,
    model: nn.Module,
    class_names: List[str],
    transform: v2.Compose,
    device: torch.device = device,
) -> str:
    """
    Predict the class label of a custom image and plot the image with the predicted label as the title.
    Args:
        - image_path: Path to the custom image
        - model: Trained PyTorch model to use for prediction
        - class_names: List of class names corresponding to the model's output labels
        - transform: torchvision.transforms to apply to the image before prediction
        - device: torch.device to perform the prediction on (default: best available device)
    Returns:    - pred_label_name: The predicted class label name for the custom image
    """

    # Load the custom image
    target_image = torchvision.io.decode_image(str(image_path)).type(torch.float32)

    # Divide by 255 to scale between 0 and 1
    target_image = target_image / 255.0

    # Apply the necessary transformations
    if transform is not None:
        target_image = transform(target_image)

    # Make model on target device
    target_image = target_image.unsqueeze(0).to(
        device
    )  # Add batch dimension and move to device
    model = model.to(device)

    # Make a prediction with the model
    model.eval()
    with torch.inference_mode():
        pred_logits = model(target_image)
        pred_label = pred_logits.argmax(dim=1).item()
        pred_label_name = class_names[pred_label]

    # Plot the image with the predicted label as the title
    plt.imshow(
        target_image.cpu().squeeze().permute(1, 2, 0)
    )  # Move to CPU, remove batch dimension and permute to (H, W, C)
    plt.title(f"Predicted: {pred_label_name}")
    plt.axis("off")

    return pred_label_name


predicted_label_name = predict_custom_image_and_plot(
    image_path=custom_image_path,
    model=model_1,
    class_names=train_dataset_augmented.classes,
    transform=custom_image_transform,
    device=device,
)

plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/f_line_223_predicted_custom_image.png"
)

print(f"Predicted label name for the custom image: {predicted_label_name}")
