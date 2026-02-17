from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch


def _to_1d_numpy(x: torch.Tensor):
    """
    Convert a tensor to a 1D NumPy array suitable for matplotlib.
    Safe for CUDA tensors and tensors requiring grad.

    Note: reshape(-1) make a tensor into flat vector, e.g., (a, b, c) shape Tensor to len(a*b*c) 1D vector
    """
    return x.detach().cpu().reshape(-1).numpy()


def plot_prediction(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    fig_save_path: (
        str | Path
    ) = "lessons/section3_pytorch_workflow/src/b_train_test_split_plot_in_function.png",
    *,
    title: str = "Train / Test Split",
    show: bool = False,
    dpi: int = 150,
) -> Path:
    """
    Plot training/test data and optionally predictions, then save the figure.

    Args:
        train_data: Training inputs (any shape that can flatten to N).
        train_labels: Training targets (same number of elements as train_data).
        test_data: Test inputs.
        test_labels: Test targets.
        predictions: Optional predictions aligned with test_data.
        fig_save_path: Where to save the figure (directories created if needed).
        title: Plot title.
        show: If True, attempt to display (may not work in headless envs).
        dpi: Saved image DPI.

    Returns:
        Path to the saved figure.
    """
    save_path = Path(fig_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x_train = _to_1d_numpy(train_data)
    y_train = _to_1d_numpy(train_labels)
    x_test = _to_1d_numpy(test_data)
    y_test = _to_1d_numpy(test_labels)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(x_train, y_train, label="Training data")
    ax.scatter(x_test, y_test, label="Test data")

    if predictions is not None:
        y_pred = _to_1d_numpy(predictions)
        ax.scatter(x_test, y_pred, label="Predictions")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)  # important for scripts / loops

    return save_path
