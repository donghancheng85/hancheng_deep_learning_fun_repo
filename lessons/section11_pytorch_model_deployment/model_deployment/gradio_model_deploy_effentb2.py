"""
Gradio demo — EfficientNet-B2 FoodVision Mini
Serves the fine-tuned EfficientNet-B2 (pizza / steak / sushi) as a local web app.

Prediction function returns:
    - Predicted label
    - Confidence dictionary  {class: probability}  rendered as a gr.Label bar chart
    - Inference time in milliseconds
"""

import timeit
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from lessons.section11_pytorch_model_deployment.model_training.effnet_b2_model_creater import (
    create_effnet_b2_model,
)
from common.device import get_best_device

# ── Config ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    "lessons/section11_pytorch_model_deployment/models/effnet_b2/"
    "effnet_b2_pizza_steak_sushi_2026-06-14_11-26-23.pth"
)
TEST_DIR = Path(
    "lessons/section6_pytorch_custom_datasets/data/pizza_steak_sushi_increased/test"
)
CLASS_NAMES = ["pizza", "steak", "sushi"]  # sorted alphabetically — matches DataLoader

# ── Load model once at startup ────────────────────────────────────────────────────────
device = get_best_device()
model, transform = create_effnet_b2_model(num_classes=len(CLASS_NAMES), seed=42)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"[INFO] Model loaded from {MODEL_PATH} → running on {device}")


# ── Prediction function ───────────────────────────────────────────────────────────────
def predict(image: Image.Image) -> tuple[str, dict[str, float], str]:
    """Run inference on a PIL image.

    Args:
        image: PIL Image supplied by the Gradio interface.

    Returns:
        A tuple of:
            - pred_label : str   — the predicted class name
            - confidences: dict  — {class_name: probability} for all classes
                                    (rendered as a label/bar chart by gr.Label)
            - time_str   : str   — inference time formatted as "X.XX ms"
    """
    # Transform → add batch dim → move to device
    img_tensor = transform(image).unsqueeze(0).to(device)

    start = timeit.default_timer()
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    elapsed_ms = (timeit.default_timer() - start) * 1000

    pred_idx = int(probs.argmax().item())
    pred_label = CLASS_NAMES[pred_idx]
    confidences = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
    time_str = f"{elapsed_ms:.2f} ms"

    return pred_label, confidences, time_str


# ── Example images (one per class) ───────────────────────────────────────────────────
EXAMPLES = [
    [str(TEST_DIR / "pizza" / "1001116.jpg")],
    [str(TEST_DIR / "steak" / "100274.jpg")],
    [str(TEST_DIR / "sushi" / "1063878.jpg")],
]

# ── Gradio interface ──────────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a food image"),
    outputs=[
        gr.Label(label="Predicted class"),
        gr.Label(num_top_classes=len(CLASS_NAMES), label="Confidence scores"),
        gr.Textbox(label="Inference time"),
    ],
    examples=EXAMPLES,
    title="FoodVision Mini — EfficientNet-B2",
    description=(
        "Upload a photo of **pizza**, **steak**, or **sushi** and the model will "
        "predict the food class along with per-class confidence scores."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
