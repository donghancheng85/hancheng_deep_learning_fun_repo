import torch

# import matplotlib
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# start with step 1: data
# data can be almost anything but in deep learning we mostly deal with tensors
# Machine learning has two pars:
# 1. get data into a numerical representation
# 2. Build a model to learn patterns in that numerical representation

# Create some linear regression formula Y = aX + b, using known parameters (model learns parameters)

# Create known parameters
weight = 0.7
bias = 0.3

"""
Create range numbers
"""
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # a matrix
y = weight * X + bias

print("first 10 element of X:")
print(X[:10])  # means get elements of first 10 rows, and all columns
print("\nfirst 10 element of y:")
print(y[:10])
print(f"length of X {len(X)} and length of y {len(y)}")

"""
Spliting data into training and test sets (very importand concept of machine learning)
1. training set (always)
2. validation set (tune model, often but not always)
3. test set (see if model if ready for application, always)
Generalization -- The ability for a machine learning model to perform well on data it has not seen before
"""
# training/test split
train_split = int(0.8 * len(X))  # 80% of X will be training
print(f"\ntraining set number {train_split}")
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(f"X training set length {len(X_train)}")
print(f"X test set length {len(X_test)}")
print(f"y training set length {len(y_train)}")
print(f"y test set length {len(y_test)}")
print(f"X shape is {X.shape}")

"""
Visualize data
"""
plt.figure(figsize=(8, 5))

# Training data
plt.scatter(
    X_train.squeeze().numpy(),  # need to squeeze because matplotlib need 1D numpy array
    y_train.squeeze().numpy(),
    color="blue",
    label="Training data",
)

# Test data
plt.scatter(
    X_test.squeeze().numpy(), y_test.squeeze().numpy(), color="red", label="Test data"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Train / Test Split Visualization")
plt.legend()
plt.grid(True)

# plt.show()
plt.savefig(
    "lessons/section3_pytorch_workflow/src/b_train_test_split.png"
)  # best practice is to store the figure
