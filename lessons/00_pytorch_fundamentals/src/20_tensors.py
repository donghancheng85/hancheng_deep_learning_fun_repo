import torch

# scalar
scalar = torch.tensor(7)
print(scalar)

dim = scalar.ndim
print(f"dim of scalar is {dim}")
int_value = scalar.item()
print(f"converted int value from scalar is {int_value}")
print(f"scalar has not type change {type(scalar)}")

print("==================================")
# vector
vector = torch.tensor([7, 7])
print(f"created vector is {vector}")
print(type(vector.ndim))
print(f"ndim of vector is {vector.ndim}")
print(f"vector.shape is {vector.shape}")

print("==================================")
# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(f"created MATRIX is \n {MATRIX}")
print(f"MATRIX ndim is {MATRIX.ndim}")
print(f"MATRIX[0] is {MATRIX[0]}")
print(f"MATRIX.shape is {MATRIX.shape}")

print("==================================")
# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
print(f"created TENSOR is \n {TENSOR}")
print(f"TENSOR.ndim is {TENSOR.ndim}")
print(f"TENSOR.shape is {TENSOR.shape}")

print("==================================")
TENSOR_1 = torch.tensor([[[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [2, 2, 2, 2],],
                          [[3 ,3 ,3, 3],
                           [3 ,2 ,3, 3],
                           [3 ,4 ,3, 3]]])

print(f"created TENSOR_1 is {TENSOR_1}")
print(f"TENSOR_1.ndim is {TENSOR_1.ndim}")
print(f"TENSOR_1.shape is {TENSOR_1.shape}")
print(f"TENSOR_1[0] is {TENSOR_1[0]}")
print(f"TENSOR_1[1][0] is {TENSOR_1[1][0]}")
