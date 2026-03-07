import torch
import matplotlib.pyplot as plt

"""
6. Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.
Feel free to reference the ML cheatsheet website for the formula.

formula:
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
"""


def self_define_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Self defined tanh funnction
    """
    exp_x = torch.exp(x)
    exp_negtive_x = torch.exp(-x)
    num = exp_x - exp_negtive_x
    dev = exp_x + exp_negtive_x
    return torch.div(num, dev)


# Visulaize and verify
x = torch.arange(start=-10, end=10, step=0.1)

y_self_define = self_define_tanh(x)
y_pytorch = torch.tanh(x)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), y_self_define.numpy())
plt.title("Self defined tanh")
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(x.numpy(), y_pytorch.numpy())
plt.title("PyTorch tanh")
plt.grid()
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/g_line_37_tanh_replicate.png"
)
