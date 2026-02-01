import torch
from torch import (
    nn,
)  # nn contains all PyTorch building blocks for neural network (graphs)
from matplotlib.pyplot import acorr  # visualization

# PyTorch Work Flow
covered_in_this_section = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "make predications and evaluting a model (inference)",
    5: "saving and loading a model",
    6: "put everyting together",
}

print(covered_in_this_section)

print(f"torch version is {torch.__version__}")

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("matmul ok, mean:", y.mean().item())
