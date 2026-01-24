import torch

"""
Reshaping, stacking, squeezing and unsqueezing tensors
* Reshaping - reshapes and input tensor to a defined shape **
* View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor **
* Stacking - combine mutiple tensors on top of each other (vstack and hstack) **
* Squeeze - remove all "1" dimensions from a tensor
* Unsqueeze - add a "1" dimension to a target tensor
* Permute - Return a view of the input with dimensions permuted (swapped) in a certain way
"""

# create a tensor and manipulate it
x = torch.arange(1.0, 10.0)  # a torch.Size([9]) tensor
print("Created tensor x = ")
print(x)
print(f"x shape is {x.shape}")

# add a extra dimension
# Note: reshape need so satisfy the original shape of the tensor
# be careful of the total number!
# in this case tensor x has 9 element, x.reshape(2, 3) will not work
# because torch.Size([2, 3]) only has 6 element, less than 9 in original
x_reshaped_add_dim0 = x.reshape(1, 9)
print("reshaped add dim0 tensor")
print(x_reshaped_add_dim0)
print(
    f"reshaped add dim0 tensor x_reshapred_add_dim0 shape is {x_reshaped_add_dim0.shape}"
)

x_reshaped_add_dim1 = x.reshape(9, 1)
print("reshaped add dim1 tensor")
print(x_reshaped_add_dim1)
print(
    f"reshaped add dim1 tensor x_reshapred_add_dim1 shape is {x_reshaped_add_dim1.shape}"
)

print("===================================")

# change the view
x_change_view = x.view(
    1, 9
)  # Note: x_change_view share the same memory of x (shallow copy)
print("changed view of x")
print(x_change_view)
print(f"x_change_view shape is {x_change_view.shape}")

# change x_change_view will change x
x_change_view[0, 0] = 5  # row 0, column 0
print("after change x_change_view, x will also be changed")
print(f"x_change_view = {x_change_view}")
print(f"x = {x}")

print("===================================")

# stack tensors on top of each other
x_stacked_dim0 = torch.stack(tensors=[x, x, x, x], dim=0)
print("Stacked tensor on dim0 (original shape is torch.Size([9]))")
print(x_stacked_dim0)
print(f"x_stacked_dim0.shape = {x_stacked_dim0.shape}")

x_stacked_dim1 = torch.stack(tensors=[x, x, x, x], dim=1)
print("Stacked tensor on dim0 (original shape is torch.Size([9]))")
print(x_stacked_dim1)
print(f"x_stacked.shape = {x_stacked_dim1.shape}")
