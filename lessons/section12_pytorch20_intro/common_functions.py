import time
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from common.device import get_best_device


def create_model(num_classes: int = 10) -> nn.Module:
    """Create a ResNet50 with ImageNet pretrained weights and a custom head.

    The final FC layer is replaced to match num_classes.
    Model is returned on CPU — move to device externally.
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_compile_model(model: nn.Module) -> tuple[nn.Module, float, torch.device]:
    """Move model to the best available device, then wrap with torch.compile().

    Note: torch.compile() is lazy — actual JIT compilation happens on
    the first forward pass, not here. The returned elapsed time covers
    only the torch.compile() setup call.

    Args:
        model: an nn.Module (on any device)

    Returns:
        compiled_model: the torch.compile-wrapped model (on best device)
        elapsed_s:      seconds taken by the torch.compile() call
        device:         the device the model was moved to
    """
    device = get_best_device()
    model = model.to(device)
    print(f"Model moved to: {device}")

    start = time.perf_counter()
    compiled_model = torch.compile(model)
    elapsed_s = time.perf_counter() - start
    print(
        f"torch.compile() setup: {elapsed_s:.4f}s  "
        f"(JIT compilation is deferred to the first forward pass)"
    )
    return compiled_model, elapsed_s, device
