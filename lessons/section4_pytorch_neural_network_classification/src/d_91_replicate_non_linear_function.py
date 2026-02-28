import torch
import matplotlib.pyplot as plt

"""
7. Replicating non-linear activation functions
Neural networks, rather than us telling the mode what to learn, we give it the tool to discover patterns in data and it tries to
figures out the patterns on its own.
And these tools are linear and non-linear functions
"""
# Create a tensor
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print(f"A shape is {A.shape}")

# Visualize the data
plt.figure(figsize=(6, 6))
plt.plot(A)
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/d_line_19_self_create_nonlinear_origin_data.png"
)


# replicate Relu
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))


print(relu(A))

plt.figure(figsize=(6, 6))
plt.plot(relu(A))
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/d_line_30_self_create_nonlinear_relu_data.png"
)


# Replicate sigmoid
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


plt.figure(figsize=(6, 6))
plt.plot(sigmoid(A))
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/d_line_30_self_create_nonlinear_sigmoid_data.png"
)
