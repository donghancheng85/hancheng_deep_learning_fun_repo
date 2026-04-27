from pathlib import Path
import random

import torch
from torch import tensor
import torchvision

# Continue with regular imports
import matplotlib.pyplot as plt

from torchmetrics import ConfusionMatrix

# Try to get torchinfo, install it if it doesn't work

# Import the going_modular directory, download it from GitHub if it doesn't work

from common.device import get_best_device, print_device_info
from common.helper_fucntion import pred_and_plot_image

from mlxtend.plotting import plot_confusion_matrix

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

# Get auto transforms matching what the model was trained with
auto_transforms = weights.transforms()

"""
1. Make predictions on the entire test dataset and plot a confusion matrix for the results of our model 
compared to the truth labels.
"""
# Setup paths and class names from the test set directory structure
data_path = Path("going_modular/data/")
test_dir = data_path / "pizza_steak_sushi" / "test"
class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

# Create lists to hold true and predicted labels
true_labels = []
pred_labels = []
pred_probs = []
img_paths = []
# Loop through each class directory in the test set
for class_dir in test_dir.iterdir():
    if class_dir.is_dir():
        # Loop through each image in the class directory
        for img_path in class_dir.iterdir():
            if img_path.is_file():
                # Make a prediction on the image and get the predicted label
                target_image = torchvision.io.read_image(str(img_path)).type(
                    torch.float32
                )
                target_image = target_image / 255.0  # scale to [0, 1]
                model.to(device)
                target_image = auto_transforms(target_image).unsqueeze(0).to(device)

                with torch.inference_mode():
                    pred_logits = model(target_image)
                    pred_label = class_names[pred_logits.argmax(dim=1).item()]
                    pred_prob = torch.softmax(pred_logits, dim=1).max().item()

                # Append the true label and predicted label to the lists
                true_labels.append(class_names.index(class_dir.name))
                pred_labels.append(class_names.index(pred_label))
                pred_probs.append(pred_prob)
                img_paths.append(img_path)

# Create a confusion matrix
confusion_matrix_calculator = ConfusionMatrix(
    task="multiclass", num_classes=len(class_names)
)
confusion_matrix = confusion_matrix_calculator(tensor(pred_labels), tensor(true_labels))
# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confusion_matrix.numpy(), class_names=class_names, figsize=(10, 7)
)
plt.savefig(
    "lessons/section8_pytorch_transfer_learning/src/c_line_92_confusion_matrix.png"
)


"""
2. Get the "most wrong" of the predictions on the test dataset and plot the 5 "most wrong" images. You can do this by:
- Predicting across all of the test dataset, storing the labels and predicted probabilities.
- Sort the predictions by wrong prediction and then descending predicted probabilities, this will give you the wrong predictions with the highest prediction probabilities, in other words, the "most wrong".
- Plot the top 5 "most wrong" images, why do you think the model got these wrong?
"""
# Collect wrong predictions and sort by descending predicted probability
wrong_predictions = [
    (img_paths[i], true_labels[i], pred_labels[i], pred_probs[i])
    for i in range(len(true_labels))
    if true_labels[i] != pred_labels[i]
]
# Sort by descending predicted probability (most confidently wrong first)
wrong_predictions.sort(key=lambda x: x[3], reverse=True)
top5_wrong = wrong_predictions[:5]

# Plot top 5 most wrong predictions
fig, axes = plt.subplots(1, len(top5_wrong), figsize=(4 * len(top5_wrong), 5))
for i, (img_path, true_idx, pred_idx, prob) in enumerate(top5_wrong):
    ax = axes[i] if len(top5_wrong) > 1 else axes
    pred_and_plot_image(
        model=model,
        image_path=img_path,
        class_names=class_names,
        transform=auto_transforms,
        device=device,
        ax=ax,
        true_class_name=class_names[true_idx],
    )
plt.suptitle("Top 5 Most Wrong Predictions (highest confidence wrong)", y=1.02)
plt.tight_layout()
plt.savefig(
    "lessons/section8_pytorch_transfer_learning/src/c_line_128_most_wrong.png",
    bbox_inches="tight",
)
plt.close()

"""
3. Predict on your own image of pizza/steak/sushi 
- how does the model go? What happens if you predict on an image that isn't pizza/steak/sushi?
"""
web_pizza_image_path = "lessons/section6_pytorch_custom_datasets/data/web_pizza_pic.jpg"
pred_and_plot_image(
    model=model,
    image_path=web_pizza_image_path,
    class_names=class_names,
    transform=auto_transforms,
    device=device,
    ax=None,
    true_class_name="Pizza (web image)",
)

plt.savefig(
    "lessons/section8_pytorch_transfer_learning/src/c_line_153_web_pizza_prediction.png"
)
plt.close()

waffle_image_path = "lessons/section6_pytorch_custom_datasets/data/web_waffle_pic.jpg"
pred_and_plot_image(
    model=model,
    image_path=waffle_image_path,
    class_names=class_names,
    transform=auto_transforms,
    device=device,
    ax=None,
    true_class_name="Waffle (web image)",
)
plt.savefig(
    "lessons/section8_pytorch_transfer_learning/src/c_line_164_web_waffle_prediction.png"
)
plt.close()
