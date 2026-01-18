import torch

"""
Note: Three big errors with PyTorch and deep learning
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on the right device
"""

# default dtype of tensor is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None)

print(f"data type of float_32_tensor is {float_32_tensor.dtype}")

# different dtype
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(f"data type of float_16_tensor is {float_16_tensor.dtype}")

# using device, grad
float_tensor = torch.tensor(
    [3.0, 6.0, 9.0],
    dtype=None,  # tensor data type
    device=None,  # What device the tensor is on, default "cpu", can also be "cuda" (Nvidia), "mps" (Apple)
    requires_grad=False,  # whether or not track the gradients with this tensor operations
)

# "convert" the datatype of a tensor and store the result to another tensor
float_16_tensor_new = float_32_tensor.type(
    torch.float16
)  # one approach to fix the dtype error of tensors
print(f"float_16_tensor_new data type is {float_16_tensor_new.dtype}")
print(f"float_32_tensor data type is still {float_32_tensor.dtype}")

# upper cast when mutilple different dtype tensors
different_dtype_tensor_multiple_result = float_16_tensor_new * float_32_tensor
print(
    f"different_dtype_tensor_multiple_result = {different_dtype_tensor_multiple_result}"
)
print(
    f"different_dtype_tensor_multiple_result.dtype = {different_dtype_tensor_multiple_result.dtype}"
)

# try int32 * float32, cast to float 32
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
int_float_multiple_result = int_32_tensor * float_32_tensor
print(f"int_float_multiple_result = {int_float_multiple_result}")
print(f"int_float_multiple_result.dtype = {int_float_multiple_result.dtype}")
