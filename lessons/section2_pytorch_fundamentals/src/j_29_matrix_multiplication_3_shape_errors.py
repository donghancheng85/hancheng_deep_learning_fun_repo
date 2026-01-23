import torch

# Dealing shape errors in matrix multiple

tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])

print(f"tensor_A shape is {tensor_A.shape}")

tensor_B = torch.tensor([[7, 10], [8, 11], [9, 12]])

print(f"tensor_B shape is {tensor_B.shape}")

# Below will not work since the innter dim does not match
# A_B_multiple_result = torch.matmul(tensor_A, tensor_B)

# Transpose, manipulate the shape of tensors, can deal with of shape errors
print(f"The transpose of tensor_B is \n {tensor_B.T}")
print(f"The shape of tensor_B transpose is {tensor_B.T.shape}")

A_BT_multiple_result = torch.matmul(tensor_A, tensor_B.T)
print(f"tensor_A @ tensor_B.T (transpose) = \n {A_BT_multiple_result}")
print(f"tensor_A @ tensor_B.T (transpose) shape is {A_BT_multiple_result.shape}")
