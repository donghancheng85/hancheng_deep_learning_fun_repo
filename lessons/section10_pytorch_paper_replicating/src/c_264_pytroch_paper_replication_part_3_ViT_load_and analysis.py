import torch
from pathlib import Path
from torch import nn

from common.device import get_best_device, print_device_info

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── ViT-Base Hyperparameters (must match the saved model exactly) ─────────────────────
IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 3
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
MLP_RATIO = 4
NUM_CLASSES = 3  # pizza / steak / sushi
DROPOUT = 0.1

CLASS_NAMES = ["pizza", "steak", "sushi"]


# ── Model Definition (identical to b_260) ─────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    """Equation 1 — converts image [B,C,H,W] → token sequence [B, N+1, D]."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        img_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # e.g. (224//16)² = 196

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  # 14×14 → 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True
        )

    def forward(self, x):  # x: [B, C, H, W]
        B = x.shape[0]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = self.flatten(x)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]
        x = x + self.pos_embed  # [B, N+1, embed_dim]  ← z₀ (Eq.1)
        return x


class MSABlock(nn.Module):
    """Equation 2 — Multi-Head Self-Attention with Pre-LN and residual."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):  # x: [B, N+1, D]
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        return attn_out + x


class MLPBlock(nn.Module):
    """Equation 3 — Feed-Forward MLP with Pre-LN and residual."""

    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: [B, N+1, D]
        return self.mlp(self.norm(x)) + x


class TransformerEncoderBlock(nn.Module):
    """One complete Transformer Encoder block = Eq.2 (MSA) + Eq.3 (MLP)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.msa_block = MSABlock(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mlp_block = MLPBlock(
            embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout
        )

    def forward(self, x):
        x = self.msa_block(x)  # Equation 2
        x = self.mlp_block(x)  # Equation 3
        return x


class ViT(nn.Module):
    """Full Vision Transformer (ViT-Base) — Equations 1–4."""

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

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
        )
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.patch_embed(x)  # Eq.1  → [B, N+1, D]
        x = self.encoder(x)  # Eq.2+3 × L → [B, N+1, D]
        cls_out = self.norm(x[:, 0])  # Eq.4  → [B, D]
        logits = self.classifier(cls_out)  #        → [B, num_classes]
        return logits


# ── Load saved weights ────────────────────────────────────────────────────────────────
model_path = Path(
    "lessons/section10_pytorch_paper_replicating/models/vit_pizza_steak_sushi_model_1_training_raw.pth"
)

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
)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

print(f"\nLoaded model from: {model_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Class names: {CLASS_NAMES}")
