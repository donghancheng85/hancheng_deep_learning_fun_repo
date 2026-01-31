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
print("============================")


# 6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this).
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
def device_agnostic_manual_seed(random_seed: int) -> None:
    """
    a device agnostic manual seed setting, if cuda, use torch.cuda.manual_seed()
    if others (mps, cpu), use torch.manual_seed()

    :param random_seed: random see to be set
    :type random_seed: int
    """
    if best_device == "cuda":
        torch.cuda.manual_seed(random_seed)
    else:  # mps or cpu
        torch.manual_seed(random_seed)


print("\nQ6")
RAND_SEED_GPU = 1234
device_agnostic_manual_seed(RAND_SEED_GPU)
rand_tensor_q6_1 = torch.rand(size=[2, 3], dtype=torch.float32, device=best_device)
print("created first 2*3 random tensor is")
print(rand_tensor_q6_1)

device_agnostic_manual_seed(RAND_SEED_GPU)
rand_tensor_q6_2 = torch.rand(size=[2, 3], dtype=torch.float32, device=best_device)
print("created second 2*3 random tensor is")
print(rand_tensor_q6_2)

print("============================")

# 7. Perform a matrix multiplication on the tensors you created in 6
# (again, you may have to adjust the shapes of one of the tensors).
print("\nQ7")

# need to transpose second tensor to perform matrix multiplication
rand_tensor_q6_2_transpose = rand_tensor_q6_2.transpose(dim0=0, dim1=1)
rand_tensor_q6_multiple_result = torch.matmul(
    rand_tensor_q6_1, rand_tensor_q6_2_transpose
)
print("two rand tensors in Q6 matrix multiple result is")
print(rand_tensor_q6_multiple_result)
print(
    f"Q6 tensors multiple result size/shape is {rand_tensor_q6_multiple_result.shape}"
)

print("============================")

# 8. Find the maximum and minimum values of the output of 7.
print("\nQ8")
max_value_Q7_mul_result = rand_tensor_q6_multiple_result.max()
print(f"find max value of Q7 result is {max_value_Q7_mul_result}")

max_value_Q7_mul_result_dim_0 = rand_tensor_q6_multiple_result.max(
    dim=0
)  # collapse on row
print(
    f"find max value of Q7 result on dim 0 (collapse by row) is {max_value_Q7_mul_result_dim_0}"
)

min_value_Q7_mul_result = rand_tensor_q6_multiple_result.min()
print(f"find min value of Q7 result is {min_value_Q7_mul_result}")

print("============================")

# 9. Find the maximum and minimum index values of the output of 7.
print("\nQ9")
max_value_Q7_mul_result_index = (
    rand_tensor_q6_multiple_result.argmax()
)  # "flatten" the 2D tensor and return the index of the max
print(f"max_value_Q7_mul_result_index = {max_value_Q7_mul_result_index}")

min_value_Q7_mul_result_index = rand_tensor_q6_multiple_result.argmin()
print(f"min_value_Q7_mul_result_index = {min_value_Q7_mul_result_index}")

print("============================")

# 10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor
# with all the 1 dimensions removed to be left with a tensor of shape (10).
# Set the seed to 7 when you create it and print out the first tensor
# and it's shape as well as the second tensor and it's shape.
print("\nQ10")
RAND_SEED_Q10 = 7
rand_tensor_q10 = torch.rand(size=[1, 1, 1, 10], dtype=torch.float32)
print("created [1, 1, 1, 10] tensor is")
print(rand_tensor_q10)
print(f"created [1, 1, 1, 10] tensor shape is {rand_tensor_q10.shape}")

rand_tensor_q10_squeezed = rand_tensor_q10.squeeze()
print("After squeese, above tensor becomees:")
print(rand_tensor_q10_squeezed)
print("After squeeze, the shape becomes")
print(rand_tensor_q10_squeezed.shape)
