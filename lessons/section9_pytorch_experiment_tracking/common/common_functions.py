import torch
import torchvision

from torch import nn


"""
7.5 Create functions to create efficientnet_b0 and efficientnet_b2 models with pretrained weights and custom classifier heads
And the models should be returned ready to be trained, with a custom classifier head and the base layers frozen.
To avoid retraining the model from pervious experiments.
"""


def create_efficientnet_b2_model(num_classes: int = 3) -> torch.nn.Module:
    """Creates an efficientnet_b2 model with pretrained weights and a custom classifier head.

    Args:
        num_classes (int, optional): The number of output classes for the classifier head. Defaults to 3.

    Returns:
        torch.nn.Module: An efficientnet_b2 model with pretrained weights and a custom classifier head.
    """
    # get the weights and transforms for efficientnet_b2
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

    # Create the model with pretrained weights
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze the base layers of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head with a custom one (dropout + linear layer with num_classes output features)
    in_features: int = model.classifier[-1].in_features  # type: ignore[union-attr]
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    model.__class__.__name__ = "efficientnet_b2"
    print(
        f"Created model: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    return model


def create_efficientnet_b0_model(num_classes: int = 3) -> torch.nn.Module:
    """Creates an efficientnet_b0 model with pretrained weights and a custom classifier head.

    Args:
        num_classes (int, optional): The number of output classes for the classifier head. Defaults to 3.

    Returns:
        torch.nn.Module: An efficientnet_b0 model with pretrained weights and a custom classifier head.
    """
    # get the weights and transforms for efficientnet_b0
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

    # Create the model with pretrained weights
    model = torchvision.models.efficientnet_b0(weights=weights)

    # Freeze the base layers of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head with a custom one (dropout + linear layer with num_classes output features)
    in_features: int = model.classifier[-1].in_features  # type: ignore[union-attr]
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    model.__class__.__name__ = "efficientnet_b0"
    print(
        f"Created model: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    return model


def create_efficientnet_b3_model(num_classes: int = 3) -> torch.nn.Module:
    """Creates an efficientnet_b3 model with pretrained weights and a custom classifier head.

    Args:
        num_classes (int, optional): The number of output classes for the classifier head. Defaults to 3.

    Returns:
        torch.nn.Module: An efficientnet_b3 model with pretrained weights and a custom classifier head.
    """
    # get the weights and transforms for efficientnet_b3
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT

    # Create the model with pretrained weights
    model = torchvision.models.efficientnet_b3(weights=weights)

    # Freeze the base layers of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head with a custom one (dropout + linear layer with num_classes output features)
    in_features: int = model.classifier[-1].in_features  # type: ignore[union-attr]
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    model.__class__.__name__ = "efficientnet_b3"
    print(
        f"Created model: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    return model
