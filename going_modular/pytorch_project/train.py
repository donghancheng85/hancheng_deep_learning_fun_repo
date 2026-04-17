"""
Using the created script in going_modular to train a model on the data created in going_modular/data.
"""

import torch
from torch import nn
from torchvision.transforms import v2

from timeit import default_timer as timer

from going_modular.pytorch_project.model_buillder import TinyVGG
from going_modular.pytorch_project.data_setup import create_dataloaders
from going_modular.pytorch_project.engine import train
from going_modular.pytorch_project.utils import save_model

from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn

# Set number of epochs
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Set up directory paths
train_dir = "going_modular/data/pizza_steak_sushi/train"
test_dir = "going_modular/data/pizza_steak_sushi/test"

# get best device for training
device = get_best_device()
print_device_info(device)

# Create transforms
data_transforms = v2.Compose(
    [
        v2.Resize(size=(64, 64)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# get data loaders and class names
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=data_transforms,
    test_transform=data_transforms,
    batch_size=BATCH_SIZE,
)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Recreate an instance of TinyVGG
model_0 = TinyVGG(
    in_features=3,  # number of color channels (3 for RGB)
    hidden_units=HIDDEN_UNITS,
    out_features=len(class_names),
).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Train model_0
model_0_results = train(
    model=model_0,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
print(f"[INFO] train results: {model_0_results}")

# Save the model
save_model(
    model=model_0,
    save_path="going_modular/models/",
    model_name="05_going_modular_script_mode_tinyvgg_model.pth",
)
