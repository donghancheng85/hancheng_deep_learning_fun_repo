import timeit
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from tqdm.auto import tqdm


def predict_on_images(
    paths: list[Path],
    model: nn.Module,
    transform: nn.Module,
    class_names: list[str],
    device: torch.device | str,
) -> list[dict]:
    """Run inference on a list of image paths and return per-sample prediction dicts.

    Args:
        paths: List of image file paths to run inference on.
        model: Trained PyTorch model.
        transform: Transforms to apply to each image before passing to the model.
        class_names: Ordered list of class name strings matching the model's output indices.
        device: Device to run inference on.

    Returns:
        A list of dicts, one per image, each containing:
            - "image_path"      : Path  — path to the source image
            - "ground_truth"    : str   — class inferred from the parent directory name
            - "pred_prob"       : torch.Tensor (on CPU) — softmax probabilities over all classes
            - "pred_class"      : str   — predicted class name
            - "time_s"          : float — inference time in seconds
            - "correct"         : bool  — whether pred_class == ground_truth
    """
    # 1. Prepare model once (send to device, set eval mode)
    model.to(device)
    model.eval()

    # 2. Container for results
    predictions = []

    for path in tqdm(paths, desc="Running inference on images"):
        # 3. Per-sample dict
        pred_dict: dict = {}

        # 4. Path and ground-truth class (inferred from parent directory name)
        pred_dict["image_path"] = path
        ground_truth = path.parent.name
        pred_dict["ground_truth"] = ground_truth

        # 5. Start timer
        start_time = timeit.default_timer()

        # 6. Load and transform image; add batch dim and move to device
        img = Image.open(path)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 7. Inference
        with torch.inference_mode():
            logits = model(img_tensor)
            pred_prob = torch.softmax(logits, dim=1)
            pred_label_idx = int(torch.argmax(pred_prob, dim=1).item())

        # 8. Store probabilities (on CPU) and predicted class name
        pred_dict["pred_prob"] = pred_prob.cpu()
        pred_dict["pred_class"] = class_names[pred_label_idx]

        # 9. End timer
        pred_dict["time_s"] = timeit.default_timer() - start_time

        # 10. Correctness check
        pred_dict["correct"] = pred_dict["pred_class"] == ground_truth

        predictions.append(pred_dict)

    return predictions
