import torch
import numpy
from common.device import get_best_device

"""
Putting tensors (models) on GPU
Because GPU makes computation faster!!!
"""

# create a tensor, default on CPU
tensor = torch.tensor(data=[1, 2, 3])
print(f"created tensor is {tensor}")
print(f"tensor is on {tensor.device}")

# Move tensor to GPU if avaliable (mps for mac, cuda for nvidia)
best_device = get_best_device()
tensor_on_gpu = tensor.to(device=best_device)
print(f"moved tensor is on {tensor_on_gpu.device}")

# Move tensor back to CPU (e.g., we are using NumPy, which is only on CPU)
# if tensor is on GPU, it cannot transfor it to NumPy
# tensor_on_gpu.numpy() <- this will not work
# To fix the issue above, need to move it back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(f"After move back to cpu, numpy tensor/array is {tensor_back_on_cpu}")

# The original tensor is not changed
print(f"\ntensor_on_gpu is {tensor_on_gpu}")
print(f"tensor_on_gpu device is {tensor_on_gpu.device}")
