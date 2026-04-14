import torch
from torch import nn


class TinyVGG(nn.Module):
    """A TinyVGG model for image classification with a fixed 64×64 input resolution.

    Implements the TinyVGG architecture from the CNN Explainer (poloclub.github.io/cnn-explainer).
    The flattened size fed into the classifier is computed automatically via a dummy forward
    pass, so kernel sizes and padding can be changed without manual recalculation.

    Architecture:
        conv_stack_1 → conv_stack_2 → Flatten → Linear(out_features)

    Each conv stack consists of:
        Conv2d → ReLU → Conv2d → ReLU → MaxPool2d(2×2)

    Args:
        in_features (int): Number of input channels (e.g. 3 for RGB images).
        hidden_units (int): Number of filters in each Conv2d layer.
        out_features (int): Number of output classes.
    """

    def __init__(self, in_features: int, hidden_units: int, out_features: int) -> None:
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Calculate in_features dynamically via a dummy forward pass ---
        # Best practice: instead of hardcoding the flattened size (which breaks
        # whenever you change kernel_size, padding, or input resolution),
        # run a zero tensor through the conv stacks to let PyTorch compute the
        # output shape for us automatically.
        # torch.no_grad(): skip gradient tracking — this is just a shape probe.
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_features, 64, 64
            )  # [batch=1, C, H, W] — matches Resize(64,64) in transform
            dummy = self.conv_stack_2(self.conv_stack_1(dummy))
            # dummy shape after both conv stacks: [1, hidden_units, H_out, W_out]
            linear_in_features = dummy.flatten(start_dim=1).shape[1]
            # flatten(start_dim=1): collapse all dims except batch → [1, hidden_units*H_out*W_out]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=linear_in_features,  # auto-computed above
                out_features=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_stack_2(self.conv_stack_1(x)))


class TinyVGGWithCustomImageShape(nn.Module):
    """A TinyVGG model for image classification that accepts arbitrary input resolution.

    Extends TinyVGG by accepting explicit image_height and image_width arguments instead
    of assuming a fixed 64×64 input. The flattened size fed into the classifier is computed
    automatically via a dummy forward pass, so the model works correctly for any input
    resolution (e.g. 224×224) without manual calculation.

    Architecture:
        conv_stack_1 → conv_stack_2 → Flatten → Linear(out_features)

    Each conv stack consists of:
        Conv2d → ReLU → Conv2d → ReLU → MaxPool2d(2×2)

    Args:
        in_features (int): Number of input channels (e.g. 3 for RGB images).
        hidden_units (int): Number of filters in each Conv2d layer.
        out_features (int): Number of output classes.
        image_height (int): Height of the input images in pixels.
        image_width (int): Width of the input images in pixels.
    """

    def __init__(
        self,
        in_features: int,
        hidden_units: int,
        out_features: int,
        image_height: int,
        image_width: int,
    ) -> None:
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Calculate in_features dynamically via a dummy forward pass ---
        # Best practice: instead of hardcoding the flattened size (which breaks
        # whenever you change kernel_size, padding, or input resolution),
        # run a zero tensor through the conv stacks to let PyTorch compute the
        # output shape for us automatically.
        # torch.no_grad(): skip gradient tracking — this is just a shape probe.
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_features, image_height, image_width
            )  # [batch=1, C, H, W] — matches Resize(image_height, image_width) in transform
            dummy = self.conv_stack_2(self.conv_stack_1(dummy))
            # dummy shape after both conv stacks: [1, hidden_units, H_out, W_out]
            linear_in_features = dummy.flatten(start_dim=1).shape[1]
            # flatten(start_dim=1): collapse all dims except batch → [1, hidden_units*H_out*W_out]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=linear_in_features,  # auto-computed above
                out_features=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_stack_2(self.conv_stack_1(x)))
