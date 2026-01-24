import torch

"""
Reshaping, stacking, squeezing and unsqueezing tensors
* Reshaping - reshapes and input tensor to a defined shape
* View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
* Stacking - combine mutiple tensors on top of each other (vstack and hstack)
* Squeeze - remove all "1" dimensions from a tensor **
* Unsqueeze - add a "1" dimension to a target tensor **
* Permute - Return a view of the input with dimensions permuted (swapped) in a certain way **
"""

# Create a orignal tensor
x = torch.randn(size=[1, 1, 3, 4])
print("Created Orignal tensor")
print(x)
print(f"Created original tensor shape = {x.shape}")

# Squeeze x
x_squeezed_dim0 = x.squeeze(dim=0)
print("\nx after squeezed by dim0:")
print(x_squeezed_dim0)
print(f"x_squeezed_dim0 shape {x_squeezed_dim0.shape}")

x_squeezed_dim1 = x.squeeze(dim=1)
print("\nx after squeezed by dim1:")
print(x_squeezed_dim1)
print(f"x_squeezed_dim1 shape {x_squeezed_dim1.shape}")

x_squeezed_dim_all = x.squeeze()
print("\nx after squeezed all dim:")
print(x_squeezed_dim_all)
print(f"x_squeezed_dim_all shape {x_squeezed_dim_all.shape}")

print("========================")

# create a new tensor for unsuqeeze
y = torch.randn(size=[3, 4])
print("Created Orignal tensor y:")
print(y)
print(f"Created original tensor shape = {y.shape}")

# unsqueeze y
y_unsqueeze_dim0 = y.unsqueeze(dim=0)
print("\ny after unsqueesed by dim0:")
print(y_unsqueeze_dim0)
print(f"y_unsqueeze_dim0 shape {y_unsqueeze_dim0.shape}")

y_unsqueeze_dim1 = y.unsqueeze(dim=1)
print("\ny after unsqueesed by dim1:")
print(y_unsqueeze_dim1)
print(f"y_unsqueeze_dim0 shape {y_unsqueeze_dim1.shape}")

y_unsqueeze_dim2 = y.unsqueeze(dim=2)
print("\ny after unsqueesed by dim2:")
print(y_unsqueeze_dim2)
print(f"y_unsqueeze_dim2 shape {y_unsqueeze_dim2.shape}")

print("========================")
# create a new tensor for permute (use case: image)
z = torch.randn(size=[3, 4, 2])
print("Originaal tensor z:")
print(z)
print(f"shape of tensor z = {z.shape}")

z_after_permute = z.permute(2, 0, 1)
print("\nAfter put dim0->dim1 dim1->dim2, dim2->dim0")
print(z_after_permute)
print(f"z_after_permute shape is {z_after_permute.shape}")
