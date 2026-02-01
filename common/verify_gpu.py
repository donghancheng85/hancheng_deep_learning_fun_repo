import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("arch list:", torch.cuda.get_arch_list())
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("matmul ok, mean:", y.mean().item())
