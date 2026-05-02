import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torchvision import transforms

from torchinfo import summary

from going_modular.pytorch_project import data_setup, engine, download_data
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds

"""
Pytorch experiment tracking is to help you keep track of your experiments, their configurations, and their results. 
This is important because it allows you to compare different experiments, reproduce results, and share your findings with others.
"""

# set up device
device = get_best_device()
print_device_info(device)

# Already exist in going_modular/data/pizza_steak_sushi, so no need to download again
# image_path = download_data.download_data(
#     source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
#     destination="pizza_steak_sushi",
# )
# print(image_path)

# Setup path to data folder
data_path = Path("going_modular/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

"""
2. Create datasets and DataLoaders
"""

"""
2.1 Create manual transforms

Make sure to fit the pre-trained model's expected input size (e.g. 224x224 for ResNet, 299x299 for Inception).
"""
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# resize -> to tensor -> normalize
manual_transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=manual_transform,
    test_transform=manual_transform,
    batch_size=32,
)


"""
2.2 Automatically create transforms

Make sure the custom data is transformed in the same way that the data the pre-trained model was trained on was transformed.
"""
# get a set of pre-trained weights for a model (EfficientNet_B0 in this case)
weights = (
    torchvision.models.EfficientNet_B0_Weights.DEFAULT
)  # Default is best avaliable weights
auto_transforms = weights.transforms()
print(f"Auto transforms: {auto_transforms}")

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=auto_transforms,
    test_transform=auto_transforms,
    batch_size=32,
)
print(f"Class names: {class_names}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of testing batches: {len(test_dataloader)}")


"""
2.3 Get the pre-trained model and update the final layer
"""
model_efficientnet_b0 = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Freeze the base layers of the model
for param in model_efficientnet_b0.features.parameters():
    # turn off gradients for the base layers, so they won't be updated during training
    param.requires_grad = False

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# Update the output layer (classifier) to match the number of classes in our dataset (3 in this case)
# Dynamically get in_features from the existing classifier's Linear layer
in_features = model_efficientnet_b0.classifier[-1].in_features
model_efficientnet_b0.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=in_features, out_features=len(class_names), bias=True),  # type: ignore[union-attr]
)

print("\nAfter freezing the base layers:")
summary(
    model=model_efficientnet_b0,
    input_size=(1, 3, 224, 224),
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "trainable",
    ],
    col_width=20,
    row_settings=["var_names"],
)

"""
2.4 Train a single model and track results
"""
# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_efficientnet_b0.parameters(), lr=0.001)

# Set up a SummaryWriter to log training metrics for TensorBoard
writer = SummaryWriter("lessons/section9_pytorch_experiment_tracking/runs")

# train the model and log results
set_seeds()
results = engine.train_for_summarywriter(
    model=model_efficientnet_b0,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    device=device,
    accuracy_fn=accuracy_fn,
    writer=writer,
    loss_fn=loss_fn,
    epochs=5,
)