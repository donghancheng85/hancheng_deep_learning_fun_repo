import torch

"""
12. Create a random tensor of shape [1, 3, 64, 64] and pass it through a nn.Conv2d() layer 
    with various hyperparameter settings (these can be any settings you choose), 
    what do you notice if the kernel_size parameter goes up and down?
"""

in_tensor = torch.zeros(1, 3, 64, 64)
print(f"in_tensor shape: {in_tensor.shape}")
print(f"{'kernel_size':<15} {'out_tensor shape'}")
print("-" * 40)

# Try different kernel sizes — notice how larger kernels shrink the spatial dimensions more
# Output formula (with stride=1, padding=0):
#   H_out = H_in - kernel_size + 1
# With padding=1:
#   H_out = H_in - kernel_size + 3
for kernel_size in [1, 3, 5, 7, 11]:
    conv_2d_layer = torch.nn.Conv2d(
        in_channels=3,
        out_channels=10,
        kernel_size=kernel_size,
        stride=1,
        padding=0,  # no padding, so we can clearly see the spatial shrinkage
    )

    with torch.no_grad():
        out_tensor: torch.Tensor = conv_2d_layer(in_tensor)

    print(f"{kernel_size:<15} {tuple(out_tensor.shape)}")
