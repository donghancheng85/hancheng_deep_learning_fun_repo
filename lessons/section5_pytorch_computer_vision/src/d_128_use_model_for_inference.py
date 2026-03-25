"""
d_128_use_model_for_inference.py

Load the saved FashionMNISTModelV2 (TinyVGG) from disk and run inference
on a handful of test images, then visualise the predictions.

Key inference steps:
  1. Re-create the model class with the SAME architecture used during training.
  2. Load the saved state_dict with torch.load(..., weights_only=True).
  3. Call model.eval() — disables dropout / batchnorm training behaviour.
  4. Wrap predictions in torch.inference_mode() — disables gradient tracking,
     saving memory and speeding up the forward pass.
  5. Convert raw logits → class probabilities with torch.softmax(), then take
     the argmax to get the predicted class index.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import random

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from mlxtend.plotting import plot_confusion_matrix

import torchmetrics

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from common.device import get_best_device, print_device_info

# ---------------------------------------------------------------------------
# 1. Device
# ---------------------------------------------------------------------------
device = get_best_device()
print_device_info(device=device)

# ---------------------------------------------------------------------------
# 2. Load test data (needed to pull sample images and class names)
# ---------------------------------------------------------------------------
test_data = datasets.FashionMNIST(
    root="lessons/section5_pytorch_computer_vision/data",
    train=False,
    download=False,
    transform=ToTensor(),
)
class_names = test_data.classes  # list of 10 class name strings
BATCH_SIZE = 32

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)


# ---------------------------------------------------------------------------
# 3. Re-define FashionMNISTModelV2 architecture
#    The architecture MUST exactly match what was used when the weights were saved.
#    (torch.save only stores weights, not the class definition.)
# ---------------------------------------------------------------------------
class FashionMNISTModelV2(nn.Module):
    """TinyVGG-style CNN used in section 5 (c_120_to_x...)."""

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28×28 → 14×14
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 4. Instantiate the model and load the saved weights
# ---------------------------------------------------------------------------
MODEL_PATH = Path("lessons/section5_pytorch_computer_vision/models")
MODEL_2_PATH = MODEL_PATH / "section5_model_2_fashionMNIST.pth"

model_2 = FashionMNISTModelV2(
    input_shape=1,  # grayscale
    hidden_units=10,
    output_shape=len(class_names),  # 10 classes
)

# weights_only=True: safer load — avoids executing arbitrary pickle code
model_2.load_state_dict(torch.load(f=MODEL_2_PATH, weights_only=True))
model_2.to(device)

# Switch to eval mode: turns off dropout / batchnorm training behaviour
model_2.eval()
print(f"Model loaded from {MODEL_2_PATH} and set to eval mode.")

"""
9, Make and evaulate random prediction with best model (tinyVGG)
"""


def make_predictions(
    model: nn.Module, data: list, device: torch.device = device
) -> torch.Tensor:
    pre_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample, add a single dimension
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass
            prediction_logit: torch.Tensor = model(sample)

            # Get prediction probability (logit -> prediction probability)
            prediction_probability = torch.softmax(
                prediction_logit.squeeze(dim=0), dim=0
            )

            # Get pre_prob off GPU for further calculations
            pre_probs.append(prediction_probability.to("cpu"))

    # Stack the pred_probs to turn list into tensor
    return torch.stack(pre_probs)


# Visualize and compare
random.seed(42)
test_samples = []
test_labels = []

# take 9 random samples
for sample, lable in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(lable)

# View th first sample shape
print(f"test_sample first element shape: {test_samples[0].shape}")

# Make predictions
pred_probs = make_predictions(
    model=model_2,
    data=test_samples,
)

# View the first two prediction probabilities
print(f"First 2 prediction probs: {pred_probs[0]} | {pred_probs[1]}")

# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
print(f" Prediction random sample, classes are: {pred_classes}")

# Plot the images with classes
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for index, sample in enumerate(test_samples):
    # Subplot for each plost
    plt.subplot(nrows, ncols, index + 1)

    # Plot target image
    plt.imshow(sample.squeeze(dim=0), cmap="gray")

    # Find prediction lable in text form
    pred_label = class_names[pred_classes[index]]

    # Get the true label in text from
    truth_label = class_names[test_labels[index]]

    # Create a title for plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Compare equality between pred and truth and change color of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")  # green, means good prediction
    else:
        plt.title(title_text, fontsize=10, c="r")  # green, means good prediction
    plt.axis(False)

plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/d_line_182_visualize_random_predictions.png"
)

"""
10. Making a confusion matrix for further evaluation

A confusion matrix is a great way to evaluation classification models visually
1. Make perdiction of trained model on the test dataset
2. make confusion matrix "torchmetrics.ConfusionMatrix"
3. To plot the confusion matrix, using mlxtend.plotting.plot_confusion_matrix()
"""

# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions...."):
        # Send the data and targets to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_logits = model_2(X)

        # Turn prediction logits → predicted class index
        # argmax(dim=1): pick the class with the highest logit across the 10 classes (dim=1).
        # No need for softmax here — softmax is monotonically increasing so argmax(softmax(x)) == argmax(x).
        # Avoid dim=0 which would incorrectly normalize across the batch instead of classes.
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # Put prediction on CPU for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
y_preds_tensor = torch.cat(y_preds)
print(len(y_preds_tensor))

# 2. Setup confusion matrix instance and compare predictions to targets
confusion_matrix_calculator = torchmetrics.ConfusionMatrix(
    task="multiclass",
    num_classes=len(class_names),
)

confusion_matrix_tensor = confusion_matrix_calculator(
    preds=y_preds_tensor, target=test_data.targets
)
print(confusion_matrix_tensor)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confusion_matrix_tensor.numpy(), class_names=class_names, figsize=(10, 7)
)
plt.savefig(
    "lessons/section5_pytorch_computer_vision/src/d_line_279_plot_confusion_matrix.png"
)
