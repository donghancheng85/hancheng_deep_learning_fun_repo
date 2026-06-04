import torch
import torchvision
from pathlib import Path
from torch import nn
import matplotlib.pyplot as plt

from common.device import get_best_device, print_device_info

# ── Device ────────────────────────────────────────────────────────────────────────────
device = get_best_device()
print_device_info(device)

# ── Constants ─────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["pizza", "steak", "sushi"]
MODEL_PATH = Path(
    "lessons/section10_pytorch_paper_replicating/models/vit_b16_pretrained_pizza_steak_sushi_2026-06-02_20-16-23.pth"
)
IMAGE_PATH = Path("lessons/section6_pytorch_custom_datasets/data/web_pizza_pic.jpg")

# ── 1. Recreate the model architecture ───────────────────────────────────────────────
# Must match the architecture used when saving:
#   - torchvision.models.vit_b_16 with head replaced to 3 classes
weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
model = torchvision.models.vit_b_16(
    weights=None
)  # architecture only, no pretrained weights
model.heads.head = nn.Linear(in_features=768, out_features=len(CLASS_NAMES))

# ── 2. Load the saved weights ─────────────────────────────────────────────────────────
print(f"\nLoading model from: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("Model loaded and set to eval mode.")

# ── 3. Prepare the same transform used during training ───────────────────────────────
# weights.transforms() gives the exact same preprocessing the model was trained with:
#   Resize(256) → CenterCrop(224) → Normalize(mean, std)
pretrained_transform = weights.transforms()
print(f"\nTransform: {pretrained_transform}")

# ── 4. Load and transform the image ──────────────────────────────────────────────────
print(f"\nImage path: {IMAGE_PATH}")
raw_image = torchvision.io.decode_image(str(IMAGE_PATH))  # uint8 tensor [C, H, W]
print(f"Raw image shape: {raw_image.shape} | dtype: {raw_image.dtype}")

image_tensor = pretrained_transform(raw_image)  # float32 [C, 224, 224]
image_batch = image_tensor.unsqueeze(0).to(device)  # [1, C, 224, 224]
print(f"Transformed image shape: {image_batch.shape} | dtype: {image_batch.dtype}")

# ── 5. Run inference ──────────────────────────────────────────────────────────────────
with torch.inference_mode():
    logits = model(image_batch)  # [1, 3]
    probs = torch.softmax(logits, dim=1)  # [1, 3]
    pred_idx = probs.argmax(dim=1).item()
    pred_label = CLASS_NAMES[pred_idx]
    pred_prob = probs[0, pred_idx].item()

print(f"\n── Prediction ──────────────────────────────")
for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs[0].tolist())):
    marker = " ◄" if i == pred_idx else ""
    print(f"  {cls:<8}: {p:.4f}{marker}")
print(f"\n  Predicted class : {pred_label}")
print(f"  Confidence      : {pred_prob:.2%}")

# ── 6. Visualise ──────────────────────────────────────────────────────────────────────
# Convert raw uint8 tensor → HWC numpy for display
img_np = raw_image.permute(1, 2, 0).numpy()

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img_np)
ax.set_title(f"Predicted: {pred_label} ({pred_prob:.2%})", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.savefig(
    Path("lessons/section10_pytorch_paper_replicating/src/f_270_prediction_result.png"),
    dpi=120,
)
plt.show()
print("\nPlot saved.")
