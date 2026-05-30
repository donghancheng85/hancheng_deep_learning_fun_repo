import time
import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Data ──────────────────────────────────────────────────────────────────────────────
data_path = Path("going_modular/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 32

manual_transform = v2.Compose(
    [
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    train_transform=manual_transform,
    test_transform=manual_transform,
    batch_size=BATCH_SIZE,
)
print(f"Class names: {class_names}")
print(f"Train batches: {len(train_dataloader)} | Test batches: {len(test_dataloader)}")

# ── ViT-Base Hyperparameters ──────────────────────────────────────────────────────────
# Matching Table 1 of "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
PATCH_SIZE = 16  # P — each patch is 16×16 pixels
IN_CHANNELS = 3  # RGB
EMBED_DIM = 768  # D — hidden/embedding dimension
NUM_HEADS = 12  # number of attention heads
NUM_LAYERS = 12  # L — number of stacked Transformer Encoder blocks
MLP_RATIO = 4  # MLP hidden dim = EMBED_DIM * MLP_RATIO = 3072
NUM_CLASSES = len(class_names)  # 3 (pizza / steak / sushi)
DROPOUT = 0.1
LEARNING_RATE = (
    3e-3  # ViT paper uses a high LR with warmup; 3e-3 is a common starting point
)
EPOCHS = 10


# ── Model Definition ──────────────────────────────────────────────────────────────────
# Re-defined here so this file is self-contained and runnable independently.
# These are the exact same classes built step-by-step in part 1.


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

        # Conv2d trick: kernel=patch_size, stride=patch_size → projects each patch to embed_dim
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  # 14×14 → 196

        # Learnable class token — prepended at position 0 (Eq.1 x_class)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # Learnable positional embeddings — one per token including cls (Eq.1 E_pos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True
        )

    def forward(self, x):  # x: [B, C, H, W]
        B = x.shape[0]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = self.flatten(x)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
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

    def forward(self, x):  # x = z_{ℓ-1}: [B, N+1, D]
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        return attn_out + x  # z'_ℓ  — residual connection


class MLPBlock(nn.Module):
    """Equation 3 — Feed-Forward MLP with Pre-LN and residual."""

    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)  # 768 * 4 = 3072
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x = z'_ℓ: [B, N+1, D]
        return self.mlp(self.norm(x)) + x  # z_ℓ  — residual connection


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

    def forward(self, x):  # x: [B, N+1, D]
        x = self.msa_block(x)  # Equation 2
        x = self.mlp_block(x)  # Equation 3
        return x


class ViT(nn.Module):
    """
    Full Vision Transformer (ViT-Base) — Equations 1–4.

    Image -> PatchEmbedding (Eq.1)
          -> TransformerEncoderBlock × L (Eq.2 + Eq.3)
          -> LayerNorm + Linear head on cls token (Eq.4)
          -> class logits
    """

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

        # Equation 1
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
        )

        # Equations 2 & 3 stacked L times
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

        # Equation 4 — final LN + classification head
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.patch_embed(x)  # Eq.1  → [B, N+1, D]
        x = self.encoder(x)  # Eq.2+3 × L → [B, N+1, D]
        cls_out = self.norm(x[:, 0])  # Eq.4  → [B, D]  (class token only)
        logits = self.classifier(cls_out)  #        → [B, num_classes]
        return logits


# ── Instantiate model ─────────────────────────────────────────────────────────────────
set_seeds()
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

print(f"\nModel: {model.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss, Optimizer, Scheduler ────────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss()

# Adam with weight decay — standard for ViT fine-tuning
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.1,  # L2 regularisation — helps prevent overfitting on small datasets
)

# Cosine annealing: gradually reduces LR to 0 over the training run
# Smoother than a step decay — standard for Transformer training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,  # number of steps until LR reaches minimum
)

# ── TensorBoard Writer ───────────────────────────────────────────────────────────────
# Logs are saved under runs/section10/ViT_<timestamp>
# Launch TensorBoard: tensorboard --logdir=runs/section10
log_dir = (
    Path("lessons/section10_pytorch_paper_replicating/runs")
    / f"ViT_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
)
writer = SummaryWriter(log_dir=str(log_dir))
print(f"TensorBoard logs → {log_dir}")

# ── Training ──────────────────────────────────────────────────────────────────────────
print(f"\nTraining ViT for {EPOCHS} epochs on {device}...")
train_start = time.perf_counter()
results = engine.train_for_summarywriter(
    model=model,
    train_data_loader=train_dataloader,
    test_data_loader=test_dataloader,
    optimizer=optimizer,
    device=device,
    accuracy_fn=accuracy_fn,
    writer=writer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    scheduler=scheduler,
)
train_end = time.perf_counter()
total_seconds = train_end - train_start
print(
    f"\nTraining time: {total_seconds:.2f}s ({total_seconds / 60:.2f} min | {total_seconds / EPOCHS:.2f}s/epoch)"
)

# ── Plot Results ──────────────────────────────────────────────────────────────────────
plot_loss_curves(results)

# ── Save Model ────────────────────────────────────────────────────────────────────────
save_dir = Path("lessons/section10_pytorch_paper_replicating/models")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "vit_pizza_steak_sushi_model_1_training_raw.pth"
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")
