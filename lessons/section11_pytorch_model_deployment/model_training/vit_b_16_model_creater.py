import torch
import torchvision
from torch import nn


def create_vit_b16_model(
    num_classes: int = 3, seed: int = 42
) -> tuple[nn.Module, nn.Module]:
    """Creates a pretrained ViT-B/16 model with a custom classifier head.

    ViT-B/16 expects 224x224 inputs; the pretrained transform handles resizing.
    All backbone parameters are frozen; only the final Linear head is trainable.

    Args:
        num_classes: The number of output classes for the model's head.
        seed: The random seed for reproducibility.
    Returns:
        A tuple containing:
            - A pretrained ViT-B/16 model with a custom head (on CPU; call .to(device) on the result).
            - The pretrained transform associated with the model.
    """
    # ── 1. Load pretrained ViT-B/16 from torchvision ─────────────────────────────────
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    model = torchvision.models.vit_b_16(weights=weights)

    # Use the transform defined by the weights — guaranteed to match what the model
    # was pretrained with (resize, crop, normalisation values etc.).
    pretrained_transform = weights.transforms()

    # ── 2. Freeze all backbone parameters ────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    # ── 3. Replace the classifier head for the target number of classes ───────────────
    # Original head: model.heads.head — Linear(768 → 1000)
    print(f"\nOriginal head: {model.heads.head}")
    torch.manual_seed(seed)
    original_head = model.heads.head
    assert isinstance(original_head, nn.Linear), "Expected heads.head to be nn.Linear"
    in_features: int = original_head.in_features  # 768 for ViT-B/16
    model.heads.head = nn.Linear(in_features=in_features, out_features=num_classes)
    print(f"Modified head: {model.heads.head}")
    # New layer is trainable; all other parameters remain frozen

    return model, pretrained_transform
