import torch

# reference: https://docs.pytorch.org/docs/stable/generated/torch.rand.html#torch.rand
# Random tensors of size(3, 4)
random_tensor = torch.rand(3, 4)
print(f"created random_tensor is \n {random_tensor}")
print(f"created random_tensor.ndim is {random_tensor.ndim}")

# create a random tensor with the same shape to an image tensor
random_image_size_tensor = torch.rand(
    size=(3, 224, 224)
)  # color channel, height, width
print(f"random_image_size_tensor.ndim is {random_image_size_tensor.ndim}")
print(f"random_image_size_tensor.shape is {random_image_size_tensor.shape}")
print(f"random_image_size_tensor.size is {random_image_size_tensor.size()}")
