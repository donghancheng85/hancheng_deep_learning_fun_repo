import torch
import torchvision
from torchvision.transforms import v2

from common.device import get_best_device, print_device_info
from lessons.section6_pytorch_custom_datasets.common.common import (
    TinyVGGWithCustomImageShape,
)

from pathlib import Path
import matplotlib.pyplot as plt

"""
8. Make a prediction on your own custom image of pizza/steak/sushi (you could even download one from the internet) and share your prediction.
Does the model you trained in exercise 7 get it right?
If not, what do you think you could do to improve it?
"""

# get best device for training
device = get_best_device()
print_device_info(device)

# Set up path to a data folder
data_path = Path("lessons/section6_pytorch_custom_datasets/data")
image_path = data_path / "pizza_steak_sushi_increased"
image_to_predict = data_path / "web_pizza_pic.jpg"

# Load the model weights from d_161_pytorch_custom_dataset_model_with_augment.py
# and use it to make a prediction on a single image downloaded from the internet.
model = TinyVGGWithCustomImageShape(
    in_features=3,
    hidden_units=64,
    out_features=3,
    image_height=224,
    image_width=224,
).to(device)
model.load_state_dict(
    torch.load(
        "lessons/section6_pytorch_custom_datasets/src/j_model_tinyvgg_224.pth",
        map_location=device,
    )
)
model.eval()

# Load the image and make a prediction on it
transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
image_loaded_float = torchvision.io.decode_image(str(image_to_predict)).float() / 255.0
image_transformed = (
    transform(image_loaded_float).unsqueeze(0).to(device)
)  # Add batch dimension and move to device
with torch.inference_mode():
    prediction = model(image_transformed)
predicted_class = prediction.argmax(dim=1).item()
class_names = ["pizza", "steak", "sushi"]
print(f"Predicted class: {class_names[predicted_class]}")

# plot the image and the prediction
plt.imshow(
    image_loaded_float.permute(1, 2, 0)
)  # Convert from (C, H, W) to (H, W, C) for plotting
plt.title(f"Predicted class: {class_names[predicted_class]}")
plt.axis("off")
plt.savefig(
    "lessons/section6_pytorch_custom_datasets/src/k_line_84_prediction_on_custom_image.png"
)
# wrong prediction, the model predicted sushi instead of pizza. To improve it, we could try:
# 1. Collecting more training data for the pizza class.
# 2. Applying more data augmentation techniques to make the model more robust.
# 3. Fine-tuning the model on a larger dataset or using a pre-trained model.
