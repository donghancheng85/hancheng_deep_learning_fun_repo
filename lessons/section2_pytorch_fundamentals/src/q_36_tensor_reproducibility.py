import torch

"""
Reproducibility: trying to take random out of random

In short, how a neural network learns:
start with random numbers -> tensor operations -> update random numnber to try and make them better
-> keep on doing that until we get ideal results
"""

# To reduce the randomness in neural networks and PyTorch comes the concept of a random seed
# Essentially, the random seed does is "flavour" the randomness

# create 2 random tensors, every time we will have different numbers (because of "random")
random_tensor_A = torch.rand(size=[3, 4])
random_tensor_B = torch.rand(size=[3, 4])
print("\nrandom_tensor_A = ")
print(random_tensor_A)
print("random_tensor_B = ")
print(random_tensor_B)
print("compare each elements between them:")
print(random_tensor_A == random_tensor_B)

# Make some random but reproducible tensors
RAMDOM_SEED = 42  # set a random seed
torch.manual_seed(RAMDOM_SEED)
random_tensor_C = torch.rand(size=[3, 4])

torch.manual_seed(RAMDOM_SEED)  # set the random seed again
random_tensor_D = torch.rand(size=[3, 4])

print("\n After setting random seed..")
print("random_tensor_C = ")
print(random_tensor_C)
print("random_tensor_D = ")
print(random_tensor_D)
print("compare two new tensors:")
print(
    random_tensor_C == random_tensor_D
)  # To make the output all True, will need to manual set the seed again
