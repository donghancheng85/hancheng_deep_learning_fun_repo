import torch

# all zeros tensor
zero_tensor = torch.zeros(size=(3, 4))
print(f"created zero tensor is \n {zero_tensor}")
print(f"zero tensor data type is {zero_tensor.dtype}")

# multiple with zero tensor
random_tensor = torch.rand(3, 4)
multiple_result = zero_tensor * random_tensor
print(f"multiple result = \n  {multiple_result}")
print(f"random tensor data type is {random_tensor.dtype}")

print("=========================")
random_tensor_1 = torch.rand(4)
multiple_result_1 = zero_tensor * random_tensor_1
print(f"multiple result 1 = \n  {multiple_result_1}")

print("=========================")
# all ones tensor
ones_tensor = torch.ones(size=(3, 4))
print(f"created ones tensor is \n {ones_tensor}")
print(f"ones tensor data type is {ones_tensor.dtype}")
