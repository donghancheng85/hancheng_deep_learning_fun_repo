import torch
import torchmetrics
import torchviz

"""
This code is to test if cuda is avaiable on the machine.
"""

print("torch:", torch.__version__)
print(f"torchmetrics {torchmetrics.__version__}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("matmul ok, mean:", y.mean().item())
else:
    print("CUDA is not avaliable on this machine. Please use CPU or MPS (Apple)")
