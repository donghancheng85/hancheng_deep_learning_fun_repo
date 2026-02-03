import torch

# start with step 1: data
# data can be almost anything but in deep learning we mostly deal with tensors
# Machine learning has two pars:
# 1. get data into a numerical representation
# 2. Build a model to learn patterns in that numerical representation

# Create some linear regression formula Y = aX + b, using known parameters (model learns parameters)

# Create known parameters
weight = 0.7
bias = 0.3

# Create range numbers
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # a matrix
y = weight * X + bias

print("first 10 element of X:")
print(X[:10])  # means get elements of first 10 rows, and all columns
print("\nfirst 10 element of y:")
print(y[:10])
