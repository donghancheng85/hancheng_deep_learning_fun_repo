"""
Exercise 3 for section 7:
Create a script to predict (such as predict.py) on a target image given a file path with a saved model.
- For example, you should be able to run the command python predict.py some_image.jpeg and have a trained PyTorch model predict on the image and return its prediction.
- You may also have to write code to load in a trained model.
"""

import torch
import torchvision
from torchvision.transforms import v2
import typer
from pathlib import Path

from going_modular.pytorch_project.model_buillder import (
    TinyVGG,
    TinyVGGWithCustomImageShape,
)

predict_app = typer.Typer(help="Run inference on an image using a saved TinyVGG model.")


@predict_app.command()
def predict_cli(
    model_path: str = typer.Option(
        "lessons/section7_pytorch_going_modular/models/complex_tinyvgg_section7.pth",
        help="Path to the saved model .pth file.",
    ),
    image_path: str = typer.Option(
        "lessons/section6_pytorch_custom_datasets/data/web_pizza_pic.jpg",
        help="Path to the image to predict on.",
    ),
    class_names: str = typer.Option(
        "pizza,steak,sushi",
        help="Comma-separated list of class names in the same order as training.",
    ),
    hidden_units: int = typer.Option(
        32,
        help="Number of hidden units — must match the saved model.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g. 'cuda' or 'cpu').",
    ),
    complex_tinyvgg: bool = typer.Option(
        False,
        help="Use TinyVGGWithCustomImageShape (224*224). Must match the architecture of the saved model.",
    ),
):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Build model matching the saved checkpoint
    model: torch.nn.Module
    transform: v2.Compose
    parsed_class_names = [c.strip() for c in class_names.split(",")]

    if complex_tinyvgg:
        model = TinyVGGWithCustomImageShape(
            in_features=3,
            hidden_units=hidden_units,
            out_features=len(parsed_class_names),
            image_height=224,
            image_width=224,
        ).to(device)
        transform = v2.Compose(
            [
                v2.Resize(size=(224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        model = TinyVGG(
            in_features=3,
            hidden_units=hidden_units,
            out_features=len(parsed_class_names),
        ).to(device)
        transform = v2.Compose(
            [
                v2.Resize(size=(64, 64)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    # Load saved weights
    model_path_resolved = Path(model_path)
    if not model_path_resolved.exists():
        typer.echo(f"[ERROR] Model file not found: {model_path_resolved}", err=True)
        raise typer.Exit(code=1)

    model.load_state_dict(torch.load(model_path_resolved, map_location=device))
    model.eval()

    # Load and transform the image
    image_path_resolved = Path(image_path)
    if not image_path_resolved.exists():
        typer.echo(f"[ERROR] Image file not found: {image_path_resolved}", err=True)
        raise typer.Exit(code=1)

    image_tensor = torchvision.io.decode_image(str(image_path_resolved))
    image_transformed = transform(image_tensor).unsqueeze(0).to(device)

    # Run inference
    with torch.inference_mode():
        logits = model(image_transformed)

    predicted_idx = logits.argmax(dim=1).item()
    predicted_class = parsed_class_names[predicted_idx]
    confidence = torch.softmax(logits, dim=1)[0][predicted_idx].item()

    typer.echo(f"Predicted class : {predicted_class}")
    typer.echo(f"Confidence      : {confidence:.2%}")


if __name__ == "__main__":
    predict_app()


"""
When predicting `lessons/section6_pytorch_custom_datasets/data/web_pizza_pic.jpg`
Using the complex TinyVGG model trained in `lessons/section7_pytorch_going_modular/models/complex_tinyvgg_section7.pth` with 32 hidden units, the output is:
Results:
Predicted class : steak
Confidence      : 47.81%

This mean is the model is not good enough to predict the image correctly. To improve it, we could try:
1. Train for more epochs.
2. Use a more complex model architecture (e.g. more hidden units, more convolutional layers, etc.).
3. Collect more training data for the pizza class.
"""
