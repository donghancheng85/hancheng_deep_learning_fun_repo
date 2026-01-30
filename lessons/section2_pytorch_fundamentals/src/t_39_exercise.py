import torch
from common.device import get_best_device

# get the best device for exercise usage
best_device = get_best_device()

# 1. Doc reading for torch.Tensor and torch.cuda

# 2. Create a random tensor with shape (7, 7).
print("Q2:")
rand_tensor = torch.rand(size=[7, 7], dtype=torch.float32)
print("Created random [7, 7] tensor is")
print(rand_tensor)
print("============================")

# 3. Perform a matrix multiplication on the tensor from 2 with another random
# tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
print("\nQ3:")
rand_tensor_2 = torch.rand(size=[1, 7], dtype=torch.float32)
print("Created rand_tensor_2 is:")
print(rand_tensor_2)
print(f"rand_tensor_2 size/shape is {rand_tensor_2.shape}")
rand_tensor_2_transpose = rand_tensor_2.transpose(
    dim0=0, dim1=1
)  # swap dim0 and dim1 of the calling tensor
print(f"size/shape of rand_tensor_2_transpose is {rand_tensor_2_transpose.shape}")
tensor_multiplication_result = torch.matmul(rand_tensor, rand_tensor_2_transpose)
print("tensor multiplication result is:")
print(tensor_multiplication_result)
print(
    f"tensor multiplication result size/shape is {tensor_multiplication_result.size()}"
)  # size() (method) and shape (attribute) are the same, but size can specify dim
print("============================")

# 4. Set the random seed to 0 and do exercises 2 & 3 over again.
# need to run multiple times so we can see the generated tensors are the same on this step
print("\nQ4:")
RAND_SEED = 0
torch.manual_seed(RAND_SEED)
# for fun, put the tensor on GPU
rand_tensor_prime = torch.rand(size=[7, 7], dtype=torch.float32, device=best_device)
print("create rand_tensor_prime is:")
print(rand_tensor_prime)

torch.manual_seed(RAND_SEED)
rand_tensor_2_prime = torch.rand(size=[1, 7], dtype=torch.float32, device=best_device)
print("create rand_tensor_2_prime is")
print(rand_tensor_2_prime)
rand_tensor_2_transpose_prime = rand_tensor_2_prime.transpose(dim0=0, dim1=1)
tensor_multiplication_result_prime = torch.matmul(
    rand_tensor_prime, rand_tensor_2_transpose_prime
)
print("seed setting random tensor multiple result is:")
print(tensor_multiplication_result_prime)

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent?
# (hint: you'll need to look into the documentation for torch.cuda for this one). If there is, set the GPU random seed to 1234.

# torch.cuda.manual_seed(seed)     -- cuda single GPU random seed:
# torch.cuda.manual_seed_all(seed) -- cuda all GPUs random seed:
# torch.manual_seed(seed)          -- CPU, MPS (Apple Silicon)
