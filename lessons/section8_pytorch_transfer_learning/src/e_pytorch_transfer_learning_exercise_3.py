from pathlib import Path

import torch
import torchvision

# Continue with regular imports

from torch import nn
from timeit import default_timer as timer

# Try to get torchinfo, install it if it doesn't work
from torchinfo import summary

# Import the going_modular directory, download it from GitHub if it doesn't work
from going_modular.pytorch_project import data_setup, engine

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, plot_confusion_matrix


# set up device
device = get_best_device()
print_device_info(device)

"""
5. Try a different model from torchvision.models on the Pizza, Steak, Sushi data, how does this model perform?
You'll have to change the size of the classifier layer to suit our problem.
You may want to try an EfficientNet with a higher number than our B0, perhaps torchvision.models.efficientnet_b2()?

Data is in here: lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased
"""
# Setup path to data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data/")
image_path = data_path / "pizza_steak_sushi_increased"
train_dir = image_path / "train"
test_dir = image_path / "test"

# get the transforms from efficientnet_b2 weights
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
transforms = weights.transforms()

# Create datasets and dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=transforms,
    test_transform=transforms,
    batch_size=32,
)

# Set up efficientnet_b2 model with pretrained weights
model = torchvision.models.efficientnet_b2(weights=weights)

# Freeze the base layers of the model
for param in model.features.parameters():
    # turn off gradients for the base layers, so they won't be updated during training
    param.requires_grad = False

# Replace the classifier head with a custom one (dropout + linear layer with 3 output features for 3 classes)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3, inplace=True),
    torch.nn.Linear(in_features=1408, out_features=len(class_names), bias=True),
)
model.to(device)

# Print out a summary of the model to see the number of trainable parameters
summary(
    model=model,
    input_size=(1, 3, 260, 260),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    col_width=20,
    row_settings=["var_names"],
)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Setup training
train_start_time = timer()

results = engine.train(
    model=model,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    device=device,
    accuracy_fn=accuracy_fn,
    loss_fn=loss_fn,
    epochs=10,
)

train_end_time = timer()
train_time = train_end_time - train_start_time
print(f"[INFO] Training time: {train_time:.3f} seconds")
print(f"Results: {results}")

confusion_matrix_save_path = "lessons/section8_pytorch_transfer_learning/src/e_line_104_confusion_matrix_double_training_data.png"
plot_confusion_matrix(
    model=model,
    test_dir=str(test_dir),
    class_names=["pizza", "steak", "sushi"],
    save_path=confusion_matrix_save_path,
    transform=transforms,
    device=device,
)
