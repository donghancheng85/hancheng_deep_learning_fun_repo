# common/device.py
import torch


def get_best_device() -> torch.device:
    """
    Select the best available device in priority order:
    CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    if device.type == "cuda":
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("Using Apple Metal (MPS)")
    else:
        print("Using CPU")
