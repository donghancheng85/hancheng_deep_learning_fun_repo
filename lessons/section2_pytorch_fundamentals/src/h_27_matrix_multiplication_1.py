import torch
import time
from common.device import get_best_device

"""
Two ways of performing multiplication in Neural Networks and Deep Learning:
1. Element-wise
2. Matirx multiplication
"""


def sync_if_needed(t: torch.Tensor):
    if t.device.type == "cuda":
        torch.cuda.synchronize()
    elif t.device.type == "mps":
        torch.mps.synchronize()


# Element-wist
tensor = torch.tensor(data=(1, 2, 3), dtype=torch.float32)
element_wise_multiple_result = tensor * tensor
print(tensor, "*", tensor)
print(f"= {element_wise_multiple_result}")

# compare run time of manual method and torch buildin method
# Matrix multiplication torch buildin method
sync_if_needed(tensor)  # to sync if tensor is on GPU
start_buildin = time.perf_counter()
matrix_multiple_result = torch.matmul(tensor, tensor)
sync_if_needed(tensor)
end_buildin = time.perf_counter()
print(f"{tensor} dot_product {tensor} = {matrix_multiple_result}")
print(f"buildin elapsed time: {(end_buildin - start_buildin) * 1000:.3f} ms")

# time compare
sync_if_needed(tensor)
start_manual = time.perf_counter()
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
sync_if_needed(tensor)
end_manual = time.perf_counter()
print(f"manual matrix multiple result = {value}")
print(f"manual elapsed time: {(end_manual - start_manual) * 1000:.3f} ms")

# Torch buildin method has much better performance
