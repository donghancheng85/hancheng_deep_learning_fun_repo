from pathlib import Path
import torch


def save_model(model: torch.nn.Module, save_path: str | Path, model_name: str) -> None:
    """Saves a PyTorch model's state_dict to disk.

    Args:
        model: The PyTorch model whose state_dict will be saved.
        save_path: Destination directory path where the model will be saved.
        model_name: Name of the model file. Must end with '.pt' or '.pth'.

    Raises:
        ValueError: If model_name does not end with '.pt' or '.pth'.
    """
    if Path(model_name).suffix not in {".pt", ".pth"}:
        raise ValueError(
            f"model_name must end with '.pt' or '.pth', got '{Path(model_name).suffix}'"
        )
    save_path = Path(save_path) / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
