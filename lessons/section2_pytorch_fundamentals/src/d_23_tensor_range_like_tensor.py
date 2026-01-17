import torch

# create a range of tensors
old_range_tensor = torch.range(0, 10)  # deprecated

new_range_tensor = torch.arange(
    0, 10
)  # correct way of using range of tensor from [0, 10)
print(f"created new_range_tensor = \n {new_range_tensor}")

another_range_tensor = torch.arange(start=0, end=1000, step=77)
print(f"another_range_tensor = \n {another_range_tensor}")

# Note: torch.arange() can only create 1D tensor (ndim = 1)
# use reshape() to create multi-dim tensor after arange()

# Creating tensor like
like_tensor = torch.zeros_like(input=new_range_tensor)
print(f"like_tensor = \n {like_tensor}")
