import torch

from common.device import get_best_device, print_device_info

device = get_best_device()
print_device_info(device)

print(f"PyTorch version: {torch.__version__}")  # PyTorch version: 2.10.0+cu128
print(torch.cuda.get_device_capability(device=device)) # (12, 0)
