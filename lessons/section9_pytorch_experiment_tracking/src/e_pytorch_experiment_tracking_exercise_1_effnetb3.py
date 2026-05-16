import torch
import torchvision
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
data_path_10_percent = Path("going_modular/data/pizza_steak_sushi")
data_path_20_percent = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased"
)

# Set up training dir
train_dir_10_percent = data_path_10_percent / "train"
train_dir_20_percent = data_path_20_percent / "train"

# Set up test dir (same for both)
test_dir = data_path_10_percent / "test"

# Hyperparameters
BATCH_SIZE = 32

"""
1.Pick a larger model from torchvision.models to add to the list of experiments 
(for example, EffNetB3 or higher).

How does it perform compared to our existing models?
"""
effnetb3_transform = torchvision.models.EfficientNet_B3_Weights.DEFAULT.transforms()

train_dataloader_10_percent_b3, test_dataloader_b3, class_names = (
    data_setup.create_dataloaders(
        train_dir=str(train_dir_10_percent),
        test_dir=str(test_dir),
        train_transform=effnetb3_transform,
        test_transform=effnetb3_transform,
        batch_size=BATCH_SIZE,
    )
)
train_dataloader_20_percent_b3, _, _ = data_setup.create_dataloaders(
    train_dir=str(train_dir_20_percent),
    test_dir=str(test_dir),
    train_transform=effnetb3_transform,
    test_transform=effnetb3_transform,
    batch_size=BATCH_SIZE,
)

# Create epoch list
num_epochs = [5, 10]

train_dataloaders = {
    "data_10_percent": train_dataloader_10_percent_b3,
    "data_20_percent": train_dataloader_20_percent_b3,
}

# Keep track of experiment numbers
experiment_number = 0

# Loop each data loader name
for train_dataloader_name in ["data_10_percent", "data_20_percent"]:
    # Loop through the epochs
    for epochs in num_epochs:
        # Loop through scheduler modes: no scheduler vs. with scheduler
        for use_scheduler in [False, True]:
            scheduler_label = "with_scheduler" if use_scheduler else "no_scheduler"

            # Increment experiment number
            experiment_number += 1

            # Set the seed for each experiment to ensure reproducibility and fair comparison
            set_seeds(seed=42)

            # Select the correct dataloaders for this model (each uses its native pretrained transform)
            train_dataloader = train_dataloaders[train_dataloader_name]
            test_dataloader = test_dataloader_b3

            # Create the model (EfficientNet-B3)
            model = create_efficientnet_b3_model(num_classes=len(class_names)).to(
                device
            )

            # Create a loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Optionally create a learning rate scheduler
            # ReduceLROnPlateau reduces LR only when val loss stops improving,
            # avoiding premature decay that hurts short fine-tuning runs.
            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2
                )
                if use_scheduler
                else None
            )

            writer = create_summary_writer(
                experiment_name=train_dataloader_name,
                model_name=model.__class__.__name__,
                extra=f"{epochs}_epochs_{scheduler_label}",
                index="scheduler_comparison",
            )

            print(
                f"\n[INFO] Experiment {experiment_number}: {train_dataloader_name} | "
                f"{epochs} epochs | {scheduler_label}"
            )

            # Train the model
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
                scheduler=scheduler,
            )

            # Save the model
            utils.save_model(
                model=model,
                save_path=Path("lessons/section9_pytorch_experiment_tracking/models"),
                model_name=f"09_{model.__class__.__name__}_{train_dataloader_name}_{epochs}_epochs_{scheduler_label}.pth",
            )
