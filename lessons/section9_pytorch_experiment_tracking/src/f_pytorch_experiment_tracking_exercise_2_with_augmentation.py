import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path

from torch import nn

from going_modular.pytorch_project import data_setup, engine, utils
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, create_summary_writer

from lessons.section9_pytorch_experiment_tracking.common.common_functions import (
    create_efficientnet_b3_model,
)


# set up device
device = get_best_device()
print_device_info(device)

# Set up paths
data_path_20_percent = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)

# Set up training dir
train_dir_20_percent = data_path_20_percent / "train"

# Set up test dir (same for both)
test_dir = data_path_20_percent / "test"

# Hyperparameters
BATCH_SIZE = 32

"""
2. Introduce data augmentation to the list of experiments using the 20% pizza, steak, sushi 
training and test datasets, does this change anything?

For example, you could have one training DataLoader that uses data augmentation 
(e.g. train_dataloader_20_percent_aug and train_dataloader_20_percent_no_aug) and then compare 
the results of two of the same model types training on these two DataLoaders.

Note: You may need to alter the create_dataloaders() function to be able to take a transform 
for the training data and the testing data (because you don't need to perform data augmentation on the test data).
"""

# EfficientNet B3 default pretrained transform (no augmentation) — used for test set and no_aug train set
effnetb3_default_transform = (
    torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()
)

# Augmented training transform using v2
# Derive the input size dynamically from the pretrained transform so this stays correct
# if the weights change (e.g. crop_size is [300] for B3, [260] for B2, etc.)
effnetb3_input_size = effnetb3_default_transform.crop_size[0]
effnetb3_aug_transform = v2.Compose(
    [
        v2.Resize((effnetb3_input_size, effnetb3_input_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create dataloaders — no augmentation
train_dataloader_no_aug, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir_20_percent),
    test_dir=str(test_dir),
    train_transform=effnetb3_default_transform,
    test_transform=effnetb3_default_transform,
    batch_size=BATCH_SIZE,
)

# Create dataloaders — with augmentation (test transform stays default, no augmentation on test)
train_dataloader_aug, _, _ = data_setup.create_dataloaders(
    train_dir=str(train_dir_20_percent),
    test_dir=str(test_dir),
    train_transform=effnetb3_aug_transform,
    test_transform=effnetb3_default_transform,
    batch_size=BATCH_SIZE,
)

# Create epoch list
num_epochs = [5, 10]

train_dataloaders = {
    "no_aug": train_dataloader_no_aug,
    "with_aug": train_dataloader_aug,
}

# Keep track of experiment numbers
experiment_number = 0

for augment_label, train_dataloader in train_dataloaders.items():
    for epochs in num_epochs:
        experiment_number += 1

        set_seeds(seed=42)

        # Create EfficientNet B3 — features frozen, only classifier head trained
        model = create_efficientnet_b3_model(num_classes=len(class_names)).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        writer = create_summary_writer(
            experiment_name="data_20_percent",
            model_name=model.__class__.__name__,
            extra=f"{epochs}_epochs_{augment_label}",
            index="augment_compare",
        )

        print(
            f"\n[INFO] Experiment {experiment_number}: 20% data | "
            f"{epochs} epochs | {augment_label}"
        )

        engine.train_for_summarywriter(
            model=model,
            train_data_loader=train_dataloader,
            test_data_loader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
            accuracy_fn=accuracy_fn,
            writer=writer,
        )

        utils.save_model(
            model=model,
            save_path=Path("lessons/section9_pytorch_experiment_tracking/models"),
            model_name=f"09_{model.__class__.__name__}_20_percent_{epochs}_epochs_{augment_label}.pth",
        )
