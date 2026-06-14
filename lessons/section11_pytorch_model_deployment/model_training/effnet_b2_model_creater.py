import torch
import torchvision
from pathlib import Path
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import time

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves


def create_effnet_b2_model(
    num_classes: int = 3, seed: int = 42
) -> tuple[nn.Module, nn.Module]:
    """Creates a pretrained EfficientNet-B2 model with a custom head for the target number of classes.

    effnet_b2 input size is 288x288, so the pretrained transform will include resizing to that size.

    Args:
        num_classes: The number of output classes for the model's head.
        seed: The random seed for reproducibility.
    Returns:
        A tuple containing:
            - A pretrained EfficientNet-B2 model with a custom head (on CPU; call .to(device) on the result).
            - The pretrained transform associated with the model.
    """
    # ── 1. Load pretrained EfficientNet-B2 from torchvision ──────────────────────────────
    # EfficientNet_B2_Weights.IMAGENET1K_V1 — pretrained on ImageNet-1k (1000 classes)
    weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Use the transform defined by the weights — guaranteed to match what the model
    # was pretrained with (resize, crop, normalisation values etc.).
    # This is safer than a manual v2.Compose: if you swap weights, the transform
    # automatically updates to match the new checkpoint.
    pretrained_transform = weights.transforms()

    # set the model's feature capture layers to not require gradients (freeze the feature extractor)
    for param in model.features.parameters():
        param.requires_grad = False

    # ── 2. Replace the head with a custom output layer for our number of classes ───────────
    # EfficientNet-B2's classifier is model.classifier[1] (after the dropout layer)
    print(f"\nOriginal head: {model.classifier[1]}")
    torch.manual_seed(seed)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    print(f"Modified head: {model.classifier[1]}")
    # New layer is trainable; all other parameters remain frozen
    return model, pretrained_transform
