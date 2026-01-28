import torch
from common.device import get_best_device

"""
Running tensors and PyTorch objectes on the GPUs (and making faster computations)
GPUs = faster computation on number, CUDA + NVIDIA + PyTorch

Ways to access GPU:
1. using google colab
2. self setup with powerful PC + GPU
3. cloud service (AWS, GCP, Azure...)
"""

# check gpu access using pytorch
cuda_is_avaliable = torch.cuda.is_available()
print(f"cuda GPU avaliability on this machine: {cuda_is_avaliable}")

# device agnostic code
best_device = get_best_device()
print(f"fastest device on this machine is {best_device}")

# Count number of devices
number_of_cuda_devices = torch.cuda.device_count()
print(f"number of cuda devices on this machine are {number_of_cuda_devices}")
