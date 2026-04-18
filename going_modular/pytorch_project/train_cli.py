"""
Exercise 2 for section 7
Use CLI to train a model on the data created in going_modular/data.

Add an argument for using a different:
- Training/testing directory
- Learning rate
- Batch size
- Number of epochs to train for
- Number of hidden units in the TinyVGG model

For example, you should be able to run something similar to the following line to
train a TinyVGG model with a learning rate of 0.003 and a batch size of 64 for 20 epochs:

python train_cli.py --learning-rate 0.003 --batch-size 64 --num-epochs 20.
"""

import torch
from torch import nn
from torchvision.transforms import v2

from timeit import default_timer as timer
import typer

from going_modular.pytorch_project.model_buillder import (
    TinyVGG,
    TinyVGGWithCustomImageShape,
)
from going_modular.pytorch_project.data_setup import create_dataloaders
from going_modular.pytorch_project.engine import train
from going_modular.pytorch_project.utils import save_model

from common.helper_fucntion import accuracy_fn

# Create root level Typer app
train_app = typer.Typer(
    help="Train a TinyVGG model on the pizza, steak, sushi dataset."
)


@train_app.command()
def train_cli(
    train_dir: str = typer.Option(
        "going_modular/data/pizza_steak_sushi/train",
        help="Path to training directory.",
    ),
    test_dir: str = typer.Option(
        "going_modular/data/pizza_steak_sushi/test",
        help="Path to testing directory.",
    ),
    learning_rate: float = typer.Option(
        0.001,
        help="Learning rate for the optimizer.",
    ),
    batch_size: int = typer.Option(
        32,
        help="Batch size for training.",
    ),
    num_epochs: int = typer.Option(
        10,
        help="Number of epochs to train for.",
    ),
    hidden_units: int = typer.Option(
        32,
        help="Number of hidden units in the TinyVGG model.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (e.g. 'cuda' or 'cpu').",
    ),
    complex_tinyvgg: bool = typer.Option(
        False,
        help="Whether to use the complex version of TinyVGG (with custom image size) or the simple version (64*64 resolution).",
    ),
    save_training: bool = typer.Option(
        False,
        help="Whether to save the trained model to disk after training.",
    ),
    save_dir: str = typer.Option(
        "lessons/section7_pytorch_going_modular/models",
        help="Directory to save the trained model if --save-training is True.",
    ),
    save_name: str = typer.Option(
        "tinyvgg_model.pth",
        help="File name to save the trained model as if --save-training is True.",
    ),
):
    # The following mean and std values are commonly used for normalizing images
    # when using pretrained models like ResNet, VGG, etc. on the ImageNet dataset.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform: v2.Compose
    test_transform: v2.Compose

    if complex_tinyvgg:
        train_transform = v2.Compose(
            [
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.RandomGrayscale(p=0.1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                v2.RandomErasing(p=0.2),
            ]
        )
        test_transform = v2.Compose(
            [
                v2.Resize(size=(224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        train_transform = v2.Compose(
            [
                v2.Resize(size=(64, 64)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        test_transform = train_transform

    # Create dataloaders first so class_names is available for model construction
    train_dataloaders, test_dataloaders, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size,
    )

    model: nn.Module

    if complex_tinyvgg:
        model = TinyVGGWithCustomImageShape(
            in_features=3,
            hidden_units=hidden_units,
            out_features=len(class_names),
            image_height=224,
            image_width=224,
        ).to(device)
    else:
        model = TinyVGG(
            in_features=3,
            hidden_units=hidden_units,
            out_features=len(class_names),
        ).to(device)

    # Make the loss function and optimizer fixed this time, so we can focus on the CLI part of the code.
    loss_fn = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler | None

    if complex_tinyvgg:
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        # CosineAnnealingLR decays lr smoothly from learning_rate to near-0 over
        # training, preventing the optimizer from overfitting late epochs.
        # Much gentler than StepLR(gamma=0.1) which kills the lr too aggressively.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs, eta_min=1e-6
        )
    else:
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = None

    # Start the timer
    start_time = timer()

    # Train the model
    train_results = train(
        model=model,
        train_data_loader=train_dataloaders,
        test_data_loader=test_dataloaders,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        epochs=num_epochs,
        device=device,
        scheduler=scheduler,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")
    print(f"[INFO] train results: {train_results}")

    if save_training:
        save_model(
            model=model,
            save_path=save_dir,
            model_name=save_name,
        )


if __name__ == "__main__":
    train_app()
