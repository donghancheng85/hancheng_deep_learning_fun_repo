import torch
from common.device import get_best_device

"""
Getting information from tensors (in Python language, attributes)

Note: To resolve three big errors with PyTorch and deep learning
1. Tensors not right datatype - get datatype from a tensor, use Tensor.dtype
2. Tensors not right shape - get shape from a tensor, use Tensor.shape
3. Tensors not on the right device - get device of a tensor, use Tensor.device
"""

# To get the above info from a tensor
test_tensor = torch.rand(3, 4)
print(f"test_tensor.dtype = {test_tensor.dtype}")
print(f"test_tensor.shape = {test_tensor.shape}") # can also use test_tensor.size(), it is a function
print(f"test_tensor.device = {test_tensor.device}")

best_device_on_this_machine = get_best_device()
print(f"The best device on this machie is {best_device_on_this_machine}")

# add a line to test commit
