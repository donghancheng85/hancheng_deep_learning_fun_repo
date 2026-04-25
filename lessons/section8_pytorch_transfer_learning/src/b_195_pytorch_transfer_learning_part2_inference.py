from pathlib import Path
import random

import torch
import torchvision

# Continue with regular imports
import matplotlib.pyplot as plt


# Try to get torchinfo, install it if it doesn't work

# Import the going_modular directory, download it from GitHub if it doesn't work

from common.device import get_best_device, print_device_info
from common.helper_fucntion import pred_and_plot_image

# set up device
device = get_best_device()
print_device_info(device)

# Loade the trained model in lessons/section8_pytorch_transfer_learning/src/a_183_pytroch_transfer_learning_part1.py
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=None)  # no pretrained weights
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=3, bias=True),  # 3 classes
)
model.load_state_dict(
    torch.load(
        "lessons/section8_pytorch_transfer_learning/models/efficientnet_b0_transfer_learning.pth",
        weights_only=True,
    )
)
model.eval()


"""
8.6 Make prediction on images from the test set

Make predictions of an image and plot the image with the predicted label and the true label.

Need to make sure:
- The image is transformed in the same way as the training data (e.g. resized, normalized).
- The same data type is used (e.g. float32).
- The same device is used (e.g. CPU or GPU) for both the model and the input data.

Will create a function called pred_and_plot_image() to make a prediction on an image and plot the image with 
the predicted label and the true label.

Steps:
1. Take in an image path, model, class names, and the transform to apply to the image.
2. Open the image with PIL, 
3. apply the transform, and add a batch dimension.
4. Make sure the mode is on target device (e.g. GPU or CPU).
5. Turn on model.eval() and inference mode (torch.inference_mode())., nn.dropout() will also trun off automatically.
6. Transfrom the target image
7. Make a prediction with the model and turn the logits into prediction probabilities.
8. Get the predicted class with the highest probability and the true label.
9. Plot the image with the predicted and true labels. (add prediction probability to the title of the plot)

Use pred_and_plot_image() in common.helper_fucntion.py, and apply it to a random sample of 9 images from the test set to create a 3x3 grid of predictions.
"""
# Setup paths and class names from the test set directory structure
data_path = Path("going_modular/data/")
test_dir = data_path / "pizza_steak_sushi" / "test"
class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

# Get auto transforms matching what the model was trained with
auto_transforms = weights.transforms()

# Gather all test images with their true class names (derived from subdirectory name)
test_images = [
    (img_path, class_dir.name)
    for class_dir in test_dir.iterdir()
    if class_dir.is_dir()
    for img_path in class_dir.glob("*.jpg")
]

# Randomly sample 9 images for a 3x3 grid
# random.seed(42)
sample_images = random.sample(test_images, 9)

# Plot 3x3 grid of predictions
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, (img_path, true_class) in enumerate(sample_images):
    pred_and_plot_image(
        model=model,
        image_path=img_path,
        class_names=class_names,
        transform=auto_transforms,
        device=device,
        ax=axes[i // 3][i % 3],
        true_class_name=true_class,
    )

plt.tight_layout()
output_path = Path(__file__).parent / "b_195_line_106_predictions_grid.png"
plt.savefig(output_path)
print(f"[INFO] Saved predictions grid to: {output_path}")
plt.close()

"""
8.7 Make prediction on image from web
"""
web_image_path = "lessons/section6_pytorch_custom_datasets/data/04-pizza-dad.jpg"
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
pred_and_plot_image(
    model=model,
    image_path=web_image_path,
    class_names=class_names,
    transform=auto_transforms,
    device=device,
    ax=ax,
    true_class_name="pizza",
)
plt.tight_layout()
output_path_web_image = Path(__file__).parent / "b_195_line_123_web_pizza_img_plot.png"
plt.savefig(output_path_web_image)
print(f"[INFO] Saved predictions grid to: {output_path_web_image}")
plt.close()
