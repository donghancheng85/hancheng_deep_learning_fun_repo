import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from torch import nn

from torchinfo import summary

from going_modular.pytorch_project import data_setup, engine
from common.device import get_best_device, print_device_info
from common.helper_fucntion import accuracy_fn, set_seeds, plot_loss_curves

# set up device
device = get_best_device()
print_device_info(device)

# Setup path to data folder
data_path = Path("going_modular/data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

print(f"torchvision version: {torchvision.__version__}")

"""
1. We want to use ViT (Vision Transformer) for image classification. Using the FoodVision mini
pizza, steak, sushi
"""

"""
2. Create datasets and DataLoaders
"""
# Create Image Size
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

print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")
print(f"Number of batches in train dataloader: {len(train_dataloader)}")
print(f"Number of batches in test dataloader: {len(test_dataloader)}")

"""
3. Visualize a single image from train_dataloader
"""
images, labels = next(iter(train_dataloader))
image = images[0]  # shape: [C, H, W], float32 in [0, 1]

fig, ax = plt.subplots()
ax.imshow(image.permute(1, 2, 0))  # CHW -> HWC
ax.set_title(f"Label: {class_names[labels[0]]}")
ax.axis("off")

save_path = Path(
    "lessons/section10_pytorch_paper_replicating/src/a_line_72_train_sample_image.png"
)
# fig.savefig(save_path)
plt.close(fig)
print(f"Saved sample image to {save_path}")

"""
3. Replicate ViT (Vision Transformer) paper overview:

Breakdown into small pieces
 - Inputs -> image tensors
 - Outputs -> What comes out of the model (e.g. class probabilities or labels)
 - Layers -> Takes input and manipulates it with a function (attention)
 - Blcoks -> A collection of layers (e.g. attention + feedforward)
 - Model (architecture) -> A collection of blocks (e.g. ViT)
"""

"""
3.1 ViT (Vision Transformer) paper overview

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
Authors: Dosovitskiy et al. (Google Brain), 2020  https://arxiv.org/abs/2010.11929

Key Insight:
  - Pure Transformer architectures (originally designed for NLP) can be applied directly
    to sequences of image patches for image classification.
  - When trained on large datasets (JFT-300M, ImageNet-21k), ViT matches or surpasses
    CNN-based models (ResNet, EfficientNet) with less compute at inference.
  - Unlike CNNs, ViT has NO built-in inductive biases (no local receptive fields,
    no translation equivariance) — the model learns spatial structure purely from data.

Architecture Pipeline:
  Image (H x W x C)
    -> Split into N fixed-size patches  (each patch is P x P x C)
    -> Flatten each patch to a 1D vector of length P²·C
    -> Linear projection (patch embedding) to dimension D
    -> Prepend a learnable [class] token
    -> Add learnable positional embeddings
    -> Feed the sequence of N+1 tokens into a standard Transformer Encoder
    -> Take the [class] token output from the final layer
    -> Pass through an MLP Head for classification

  where:
    H, W = image height, width
    C    = number of channels (e.g. 3 for RGB)
    P    = patch size (e.g. 16 for ViT-16)
    N    = H·W / P²  (number of patches, becomes the sequence length)
    D    = embedding dimension (hidden size of the Transformer)

Four Core Equations from the paper (Section 3, Equation 1-4):
─────────────────────────────────────────────────────────────────────────────────────────

Equation 1 — Patch + Position Embedding (input preparation):

  z₀ = [x_class ; x_p¹·E ; x_p²·E ; ... ; x_pᴺ·E] + E_pos

  - x_pⁱ ∈ ℝ^(P²·C)         : flattened i-th image patch
  - E    ∈ ℝ^(P²·C × D)      : learnable linear projection (the patch embedding layer)
  - x_class ∈ ℝ^D             : learnable [class] token prepended at position 0
  - E_pos  ∈ ℝ^((N+1) × D)   : learnable 1D positional embeddings added element-wise
  - z₀   ∈ ℝ^((N+1) × D)     : the full input sequence to the Transformer Encoder

  Purpose: Converts a 2D image into a 1D sequence of D-dimensional tokens so the
  standard Transformer can process it. The [class] token aggregates global image
  information; positional embeddings inject spatial order (since attention is
  permutation-invariant).

─────────────────────────────────────────────────────────────────────────────────────────

Equation 2 — Multi-Head Self-Attention (MSA) sub-layer (for layer ℓ):

  z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}

  - LN               : Layer Normalization applied BEFORE attention (Pre-LN variant)
  - MSA              : Multi-Head Self-Attention — each token attends to every other token
  - + z_{ℓ-1}        : residual (skip) connection — adds the un-normalized input back
  - z'_ℓ             : intermediate output after the attention sub-layer

  Purpose: Allows every token (patch) to exchange information with every other token
  across the entire sequence in parallel. The Pre-LN + residual design stabilises
  training in deep networks.

─────────────────────────────────────────────────────────────────────────────────────────

Equation 3 — MLP (Feed-Forward) sub-layer (for layer ℓ):

  z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ

  - LN               : Layer Normalization applied BEFORE the MLP (Pre-LN variant)
  - MLP              : Two linear layers with a GELU activation in between
                       (expands D -> 4D then projects back 4D -> D)
  - + z'_ℓ           : residual (skip) connection
  - z_ℓ              : output of Transformer Encoder layer ℓ

  Purpose: After global information mixing via MSA, the MLP processes each token
  independently and non-linearly, acting as the "reasoning" step per position.
  Together, Equations 2 & 3 form one complete Transformer Encoder block.
  These blocks are stacked L times.

─────────────────────────────────────────────────────────────────────────────────────────

Equation 4 — Classification Head (final prediction):

  y = LN(z_L⁰)

  - z_L⁰             : the [class] token output from the LAST (L-th) Transformer layer
                       (index 0 of the sequence, shape: [D])
  - LN               : final Layer Normalization
  - y                : passed through a linear MLP head -> class logits

  Purpose: Only the [class] token is used for prediction — it has attended to all
  patch tokens across all L layers and therefore encodes a global image representation.
  During pre-training a 2-layer MLP head is used; during fine-tuning a single linear
  layer suffices.

─────────────────────────────────────────────────────────────────────────────────────────

Summary flow of equations:
  Image -> Eq.1 (embed patches + positions) -> [Eq.2 (MSA) + Eq.3 (MLP)] × L layers
        -> Eq.4 (extract class token) -> MLP Head -> class probabilities

═════════════════════════════════════════════════════════════════════════════════════════
PSEUDO CODE — mapping each equation to PyTorch building blocks
═════════════════════════════════════════════════════════════════════════════════════════

# ── Hyperparameters (ViT-Base defaults) ──────────────────────────────────────────────
# IMG_SIZE  = 224        # input image height = width
# PATCH_SIZE = 16        # each patch is 16×16 pixels
# IN_CHANNELS = 3        # RGB
# NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # = 196
# EMBED_DIM = 768        # D — hidden/embedding dimension
# NUM_HEADS = 12         # number of attention heads in MSA
# MLP_RATIO = 4          # MLP hidden dim = EMBED_DIM * MLP_RATIO = 3072
# NUM_LAYERS = 12        # L — number of stacked Transformer Encoder blocks
# NUM_CLASSES = 3        # output classes (pizza / steak / sushi)
# DROPOUT = 0.1

# ── Equation 1 — PatchEmbedding layer ────────────────────────────────────────────────
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels, patch_size, embed_dim):
#         # Option A — Conv2d trick: kernel_size=patch_size, stride=patch_size
#         #   projects each non-overlapping patch directly to embed_dim in one step
#         self.proj = nn.Conv2d(in_channels,
#                               embed_dim,
#                               kernel_size=patch_size,
#                               stride=patch_size)
#         # Option B — explicit flatten + Linear (more literal to the paper):
#         # self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
#
#         self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))   # x_class
#         self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # E_pos
#
#     def forward(self, x):                     # x: [B, C, H, W]
#         x = self.proj(x)                      # [B, embed_dim, H/P, W/P]
#         x = x.flatten(2).transpose(1, 2)      # [B, N, embed_dim]   (N = num_patches)
#         cls = self.cls_token.expand(B, -1, -1)# [B, 1, embed_dim]
#         x = torch.cat([cls, x], dim=1)        # [B, N+1, embed_dim]  prepend [class]
#         x = x + self.pos_embed                # [B, N+1, embed_dim]  + positional emb
#         return x                              # z₀

# ── Equation 2 — MSA sub-layer (inside one TransformerEncoderBlock) ──────────────────
# class MSABlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout):
#         self.norm = nn.LayerNorm(embed_dim)                      # LN before attention
#         self.attn = nn.MultiheadAttention(embed_dim,
#                                           num_heads,
#                                           dropout=dropout,
#                                           batch_first=True)
#
#     def forward(self, x):                     # x = z_{ℓ-1}: [B, N+1, D]
#         normed = self.norm(x)                 # LN(z_{ℓ-1})
#         attn_out, _ = self.attn(normed, normed, normed)  # MSA(...)
#         return attn_out + x                   # z'_ℓ  — residual connection

# ── Equation 3 — MLP sub-layer (inside one TransformerEncoderBlock) ──────────────────
# class MLPBlock(nn.Module):
#     def __init__(self, embed_dim, mlp_ratio, dropout):
#         hidden_dim = int(embed_dim * mlp_ratio)               # 768 * 4 = 3072
#         self.norm = nn.LayerNorm(embed_dim)                   # LN before MLP
#         self.mlp  = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.GELU(),                                        # paper uses GELU
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, x):                     # x = z'_ℓ: [B, N+1, D]
#         return self.mlp(self.norm(x)) + x     # z_ℓ  — residual connection

# ── One full Transformer Encoder Block = Eq.2 + Eq.3 ────────────────────────────────
# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
#         self.msa_block = MSABlock(embed_dim, num_heads, dropout)
#         self.mlp_block = MLPBlock(embed_dim, mlp_ratio, dropout)
#
#     def forward(self, x):
#         x = self.msa_block(x)   # Equation 2
#         x = self.mlp_block(x)   # Equation 3
#         return x

# ── Equation 4 — Classification Head ─────────────────────────────────────────────────
# class ViT(nn.Module):
#     def __init__(self, ...):
#         self.patch_embed = PatchEmbedding(...)               # Equation 1
#         self.encoder     = nn.Sequential(
#             *[TransformerEncoderBlock(...) for _ in range(NUM_LAYERS)]
#         )                                                    # Equations 2 & 3, × L
#         self.norm        = nn.LayerNorm(EMBED_DIM)           # final LN  (Eq. 4)
#         self.head        = nn.Linear(EMBED_DIM, NUM_CLASSES) # MLP Head
#
#     def forward(self, x):                     # x: [B, C, H, W]
#         x = self.patch_embed(x)               # Eq.1  -> [B, N+1, D]
#         x = self.encoder(x)                   # Eq.2+3 × L -> [B, N+1, D]
#         cls_out = self.norm(x[:, 0])          # Eq.4  -> [B, D]  (class token only)
#         logits  = self.head(cls_out)          # -> [B, NUM_CLASSES]
#         return logits
"""

"""
In table one of the ViT paper, we have different variants of the ViT architecture (ViT-Base, ViT-Large, ViT-Huge) with different hyperparameters. For this example, 
we'll be replicating the ViT-Base architecture which has the following hyperparameters:
  - Image size: 224 x 224
  - Patch size: 16 x 16
  - Embedding dimension (Hidden size): 768 (transform the 16x16 patches into 768-dimensional vectors)
  - Number of heads: 12
  - Layer depth: 12 (number of stacked Transformer Encoder blocks)
  - MLP size: 3072 (the hidden dimension of the MLP in the Transformer Encoder blocks, which is typically 4x the embedding dimension)
    MLP is a feedforward network that processes each token independently after the attention mechanism has mixed information across tokens.
"""


"""
Equation 1 — Patch + Position Embedding (input preparation):
Input shape: [B, C, H, W] -> Output shape: [B, N+1, D]
  - B: batch size
  - C: number of channels (3 for RGB)
  - H, W: image height and width (224 x 224)
  - N: number of patches = (H / P) * (W / P) = (224 / 16) * (224 / 16) = 14 * 14 = 196
  - P: patch size (16)
  - D: embedding dimension (768)
"""
# Calculate the number of patches (N) based on the image size and patch size
height = 224
width = 224
color_channels = 3
patch_size = 16
number_of_patches = (height // patch_size) * (width // patch_size)  # = 196
print(f"Number of patches (N): {number_of_patches}")

embedding_layer_input_shape = (height, width, color_channels)
embedding_layer_output_shape = (
    number_of_patches,
    patch_size**2 * color_channels,
)  # (N, D)
print(
    f"Embedding layer input shape (H, W, C), a 2D image: {embedding_layer_input_shape}"
)
print(
    f"Embedding layer output shape (N, D), single 1D sequence per patch: {embedding_layer_output_shape}"
)

"""
4.2 Turn a single image into a sequence of patches and visualize the output of the embedding layer
"""
# View a single image
images, labels = next(iter(train_dataloader))
image = images[0]  # shape: [C, H, W], float32 in [0, 1]
print(f"Original image shape: {image.shape}")
plt.imshow(image.permute(1, 2, 0))  # CHW -> HWC
plt.title(f"Original Image - Label: {class_names[labels[0]]}")
plt.axis("off")
# plt.show()
plt.close()

# Plot the patched image, using matplotlib's subplot to visualize the patches
num_patches_per_dim = height // patch_size  # 224 // 16 = 14
fig, axes = plt.subplots(num_patches_per_dim, num_patches_per_dim, figsize=(10, 10))
print(
    f"Number of patches per dimension: {num_patches_per_dim} (total patches: {num_patches_per_dim**2})"
)
for i in range(num_patches_per_dim):
    for j in range(num_patches_per_dim):
        patch = image[
            :,
            i * patch_size : (i + 1) * patch_size,
            j * patch_size : (j + 1) * patch_size,
        ]
        axes[i, j].imshow(patch.permute(1, 2, 0))  # CHW -> HWC
        axes[i, j].axis("off")
plt.suptitle("Image split into 16x16 patches")
# plt.show()
# plt.savefig(
#     "lessons/section10_pytorch_paper_replicating/src/a_line_362_image_split_into_patches.png"
# )
plt.close(fig)
print("Visualized the image split into 16x16 patches.")


"""
4.3 Understanding the patch embedding layer using a Conv2d 
to turn patches into embeddings and visualize the output shape
"""
# Create patch embedding layer using conv2d with kernel_size=patch_size and stride=patch_size
embed_dim = 768  # D — ViT-Base hidden size (design hyperparameter)

patch_proj = nn.Conv2d(
    in_channels=color_channels,  # 3 (RGB)
    out_channels=embed_dim,  # 768 filters — this IS the projection matrix E ∈ ℝ^(P²C × D)
    kernel_size=patch_size,  # 16 — each filter covers one full patch
    stride=patch_size,  # 16 — non-overlapping: jump exactly one patch at a time
)

# Pass a single image through the projection layer (add batch dim with unsqueeze)
# image shape: [C, H, W] = [3, 224, 224]
image_batch = image.unsqueeze(0)  # [1, 3, 224, 224]

# (batch size, embed_dim, num_patches_per_dim, num_patches_per_dim)
x_proj = patch_proj(image_batch)  # [1, 768, 14, 14]
print(
    f"x_proj (after Conv2d)        : {x_proj.shape}"
)  # projected spatial maps — NOT yet a sequence

# flatten dims 2+ (the spatial 14×14 grid → 196)
flatten = nn.Flatten(start_dim=2, end_dim=3)
x_flat = flatten(x_proj)  # [1, 768, 196]
print(
    f"x_flat (after nn.Flatten)    : {x_flat.shape}"
)  # spatially flattened — axes still [B, D, N]

patch_embeddings = x_flat.transpose(1, 2)  # [1, 196, 768]
print(
    f"patch_embeddings (after transpose)   : {patch_embeddings.shape}"
)  # [B, N, D] — the actual x_pⁱ·E vectors (Eq.1, before cls + pos)


# Create the patch embedding layer using a linear layer (alternative to Conv2d)
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        img_size: int = 224,
    ):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # e.g. (224 // 16)² = 196

        # ── Projection: Conv2d maps each patch → embed_dim vector (the E matrix in Eq.1)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  # collapse spatial 14×14 → 196

        # ── Class token (x_class in Eq.1)
        # A single learnable vector prepended at position 0.
        # It attends to all patch tokens across all layers and accumulates
        # global image information — only this token is used for classification (Eq.4).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # ── Positional embeddings (E_pos in Eq.1)
        # Self-attention is permutation-invariant — without this, the model cannot
        # distinguish patch position 0 (top-left) from patch position 195 (bottom-right).
        # Shape: [1, N+1, embed_dim] — one learnable vector per token (patches + cls).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)

    def forward(self, x):  # x: [B, C, H, W]
        B = x.shape[0]
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, (
            f"Input image size must be divisible by patch size, "
            f"image shape: {image_resolution}, patch size: {self.patch_size}"
        )

        x = self.proj(x)       # [B, embed_dim, H/P, W/P]   — projection (x_pⁱ · E)
        x = self.flatten(x)    # [B, embed_dim, N]            — spatial grid → flat sequence
        x = x.transpose(1, 2)  # [B, N, embed_dim]            — token-first order

        # Prepend class token: expand to batch size, then cat at sequence position 0
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)           # [B, N+1, embed_dim]

        # Add positional embeddings element-wise — injects spatial order into each token
        x = x + self.pos_embed                          # [B, N+1, embed_dim]  ← z₀ (Eq.1)
        return x

set_seeds()
patch_embedding_layer = PatchEmbedding(in_channels=color_channels, patch_size=patch_size, embed_dim=embed_dim)
patch_embeddings = patch_embedding_layer(image_batch)
print(
    f"patch_embeddings (after PatchEmbedding layer)   : {patch_embeddings.shape}"
)  # [B, N, D] — the actual x_pⁱ·E vectors (Eq.1, before cls + pos)

