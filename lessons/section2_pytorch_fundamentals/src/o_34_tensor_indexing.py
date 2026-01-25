import torch

"""
Indexing with PyTorch is similar with Numpy
"""

# create a randn torch.Size([2, 3, 4]) tensor first
# tensor dim is 3, batch, row, column
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print("Created tensor is:")
print(x)
print(f"Created tensor shape is: {x.shape}")

# dim0 index (batch index)
x_dim0_index_0 = x[0]  # batch 0
print("\n x[0] is")
print(x_dim0_index_0)
print(f"shape of x[0] is {x_dim0_index_0.shape}")
print(f"dim of x[0] is {x_dim0_index_0.dim()}")

x_dim0_index_0_another = x[0, :, :]  # another way to index batch 0
print("\n x[0, :, :] is")
print(x_dim0_index_0_another)

# dim1 index
x_dim1_index = x[0, 1, :]  # batch 0, everything in the the 1st row
# x_dim1_index = x[0][1] is the same as above
print("\n x[0, 1, :] is")
print(x_dim1_index)
print(f"x[0, 1, :] shape is {x_dim1_index.shape}")
print(f"x[0, 1, :] dim is {x_dim1_index.dim()}")

# dim2 index (a element, dim=0)
x_dim2_index = x[1, 2, 2]  # batch 1, row 2, column 2
# x_dim2_index = x[1][2][2] is the same as above
print("\n x[1, 2, 2] is")
print(x_dim2_index)
print(f"x[1, 2, 2] shape is {x_dim2_index.shape}")
print(f"x[1, 2, 2] dim is {x_dim2_index.dim()}")

# Slice ":"
x_dim1_slice = x[:, 1, :]  # all batches row 1
print("\nx[:, 1, :] is")
print(x_dim1_slice)
print(f"x[:, 1, :] shape is {x_dim1_slice.shape}")
print(f"x[:, 1, :] dim is {x_dim1_slice.dim()}")

x_dim2_slice = x[:, :, 3]  # all batches, column 3
print("\nx[:, :, 3] is")
print(x_dim2_slice)
print(f"x[:, :, 3] shape is {x_dim2_slice.shape}")
print(f"x[:, :, 3] dim is {x_dim2_slice.dim()}")
