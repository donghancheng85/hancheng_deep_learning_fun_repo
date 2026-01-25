import torch
import numpy as np

"""
NumPy is a popular scientific Python numerical computing lib
PyTorch has functionality to interact with it

* Data in NumPy, what in PyTorch tenor -> torch.from_numpy(ndarray)
* PyTorch tensor convert to NumPy -> torch.Tensor.numpy()
"""

# NumPy array to PyTorch tensor
np_array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(np_array)
print(f"NumPy array is {np_array}")
print(f"converted to PyTorch tensor is {tensor}")

# be careful of dtype, NumPy default dtype is float64, PyTorch reflects NumPy data type!!!
print(f"NumPy np_array dtype is {np_array.dtype}")
print(f"converted PyTorch tensor dtype is {tensor.dtype}")

# to update date type, we can do this
tensor_use_pytorch_default_type = torch.from_numpy(np_array).type(torch.float32)
print(f"\nusing PyTorch default dtype created tensor {tensor_use_pytorch_default_type}")
print(
    f"using PyTorch default dtype created tensor dtype is {tensor_use_pytorch_default_type.dtype}"
)

# Change value of np_array -> will that change the value of the converted tensor?
# Yes, they share the same memory
np_array[0] = 332  # update the value of np_array
# tensor = tensor + 1 -> Note: this will not change np_array value because it caused recopy of tensor so it does not point to the original memeory
print("\nAfter changing np_array value")
print(f"np_array = {np_array}")
print(f"tensor is {tensor}")

print("==========================================================")

# From torch.Tensor to numpy
torch_tensor = torch.ones(7)
numpy_array_from_torch_tensor = torch_tensor.numpy()
print(f"\nOriginal torch.Tensor is {torch_tensor}")
print(f"converted NumPy array is {numpy_array_from_torch_tensor}")
print(
    f"dtype of converted array {numpy_array_from_torch_tensor.dtype}"
)  # reflects the original tensor dtype

# The same, since torch.Tensor.numpy() created a shallow copy, change converted numpy will change the original
torch_tensor[1] = 99
# torch_tensor = torch_tensor + 1 -> again, this will not change numpy_array_from_torch_tensor
print(f"\nAfter changing, Original torch.Tensor is {torch_tensor}")
print(
    f"After changing orignal torch.Tensor converted NumPy array is {numpy_array_from_torch_tensor}"
)
