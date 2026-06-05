"""
Replicate the ViT architecture we created with in-built PyTorch transformer layers.
You'll want to look into replacing our TransformerEncoderBlock() class with torch.nn.TransformerEncoderLayer() (these contain the same layers as our custom blocks).
You can stack torch.nn.TransformerEncoderLayer()'s on top of each other with torch.nn.TransformerEncoder().

Mapping from custom classes → built-in:
  MSABlock + MLPBlock + TransformerEncoderBlock  →  nn.TransformerEncoderLayer
  nn.Sequential([TransformerEncoderBlock] * L)   →  nn.TransformerEncoder
"""

import torch
from torch import nn
from torchinfo import summary

from common.device import get_best_device, print_device_info

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── ViT-Base Hyperparameters ──────────────────────────────────────────────────────────
IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 3
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
MLP_RATIO = 4
NUM_CLASSES = 3  # pizza / steak / sushi
DROPOUT = 0.1


# ── Eq. 1: PatchEmbedding — unchanged from c_264 ──────────────────────────────────────
class PatchEmbedding(nn.Module):
    """Converts image [B, C, H, W] → token sequence [B, N+1, D]."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        img_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # (224//16)² = 196

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  # 14×14 → 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True
        )

    def forward(self, x):  # x: [B, C, H, W]
        B = x.shape[0]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = self.flatten(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed  # [B, N+1, D]  ← z₀  (Eq. 1)
        return x


# ── Eq. 2 + 3: replaced by nn.TransformerEncoderLayer ────────────────────────────────
#
# Key parameters to match our custom implementation:
#   norm_first=True   → Pre-LN (LayerNorm before attention and MLP), same as our blocks
#   activation='gelu' → matches GELU in our MLPBlock
#   batch_first=True  → input is [B, N, D], not [N, B, D]
#   dim_feedforward   → embed_dim * mlp_ratio (768 * 4 = 3072)
#
# nn.TransformerEncoder stacks N of these and optionally applies a final norm.


class ViT(nn.Module):
    """ViT-Base using nn.TransformerEncoderLayer instead of custom blocks."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Eq. 1 — patch + position embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
        )

        # Eq. 2 + 3 — single encoder layer (MSA + MLP, Pre-LN, GELU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # [B, N, D]  not  [N, B, D]
            norm_first=True,  # Pre-LN, matches our custom MSABlock / MLPBlock
        )

        # Stack L encoder layers  (replaces our nn.Sequential of TransformerEncoderBlock)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Eq. 4 — final LayerNorm + classification head
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.patch_embed(x)  # Eq. 1  → [B, N+1, D]
        x = self.encoder(x)  # Eq. 2+3 × L → [B, N+1, D]
        cls_out = self.norm(x[:, 0])  # Eq. 4  → [B, D]
        logits = self.classifier(cls_out)  #       → [B, num_classes]
        return logits


# ── Instantiate and inspect ───────────────────────────────────────────────────────────
model = ViT(
    img_size=IMAGE_SIZE,
    in_channels=IN_CHANNELS,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_ratio=MLP_RATIO,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(device)

print("\n── ViT with nn.TransformerEncoderLayer ──")
summary(
    model=model,
    input_size=(32, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)

# ── Sanity-check forward pass ─────────────────────────────────────────────────────────
dummy = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
with torch.inference_mode():
    out = model(dummy)
print(f"\nOutput shape: {out.shape}")  # expected: torch.Size([1, 3])
assert out.shape == (1, NUM_CLASSES), f"Unexpected output shape: {out.shape}"
print("Forward pass OK.")
