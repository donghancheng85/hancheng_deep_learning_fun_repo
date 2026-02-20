import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt

"""
Cliassification is a problem of predicting whether someting is one thing or another (there can be multiple things as the options)
"""

"""
1. Make classification data and get it ready
"""
# Make 1000 samples
n_samples = 1000

# Create circles (X - features, y -labels)
X, y = make_circles(
    n_samples=n_samples,
    noise=0.03,
    random_state=42,
)

print(f"Created data set length len(X)={len(X)}, len(y)={len(y)}")
print(f"First 5 element of X \n{X[:5]}, \n y {y[:5]}")

# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
print(f"Created circles DataFrame first 10 elements are \n {circles.head(10)}")

# Visualize the data
plt.figure(figsize=(10, 7))
plt.scatter(
    x=X[:, 0],
    y=X[:, 1],
    c=y,
    cmap="RdYlBu",
)
plt.savefig("lessons/section4_pytorch_neural_network_classification/src/b_visualize_generated_circle_data.png")

# Note: the data just created with is often referred to as a toy dataset, 
# a dataset that is small enough to experiment but still sizeable enough to practice the fundamentals

"""
1.1 Check input and output shapes
"""
print("\n")
print(f"Shape of X - {X.shape}, y - {y.shape}")

# View the first example of features and labels
X_sample: ndarray = X[0]
y_sample: ndarray = y[0]

print(f"Values for one sample of X: {X_sample}, y: {y_sample}")
print(f"Shape of one sample of X: {X_sample.shape}, y: {y_sample.shape}") # two features for input X, one feature for y

"""
1.2 Turn data into tensors, create train and test splits
"""
# Turn data into tensors
X = torch.from_numpy(X).type(torch.float32) # ndarray default dtype is float64
y = torch.from_numpy(y).type(torch.float32)

print(f"After convert to Tensor, X first 5 elements:\n {X[:5]}, y first 5 elements:\n {y[:5]}")

# Split into training and test dataset, using randmo approach

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
) # 20% test size, 80% train

print(
    "Training and testing data length:\n"\
    f"X_train: {len(X_train)} | X_test: {len(X_test)} | y_train: {len(y_train)} | y_test: {len(y_test)}"
)
