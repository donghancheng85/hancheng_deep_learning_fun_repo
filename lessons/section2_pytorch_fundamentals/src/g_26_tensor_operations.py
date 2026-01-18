import torch
from common.device import get_best_device

# Manipulating Tensors (tensor operations)

"""
Tensor Operations include:
1. Addition
2. Substraction
3. Mutiplication (element-wise)
4. Division
5. Matrix mutiplication
"""

tensor_for_operation = torch.tensor(data=[1., 2., 3.])
print(f"tensor_for_operation = \n {tensor_for_operation}")

# Addition
addition_result = tensor_for_operation + 10
print(f"addition_result = {addition_result}")

# Muliplication
multiple_result = tensor_for_operation * 10
print(f"multiple_result = {multiple_result}")

# Substract
substract_result = tensor_for_operation - 10
print(f"substract_result = {substract_result}")

# Use PyTorch buildin functions
buildin_fun_mul_result = torch.mul(tensor_for_operation, 10)
print(f"buildin_fun_mul_result = {buildin_fun_mul_result}")
