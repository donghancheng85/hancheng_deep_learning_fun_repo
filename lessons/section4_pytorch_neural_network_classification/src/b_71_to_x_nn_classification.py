import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt

from common.device import get_best_device, print_device_info

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
y = torch.from_numpy(y).type(torch.float32).unsqueeze(dim=1) # need to unsqueeze because the model output shape will be (feature_num, other_dim)

print(f"After convert to Tensor, X first 5 elements:\n {X[:5]}, y first 5 elements:\n {y[:5]}")

# Split into training and test dataset, using randmo approach

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
) # 20% test size, 80% train
X_train: torch.Tensor
X_test: torch.Tensor
y_train: torch.Tensor
y_test: torch.Tensor

print(
    "Training and testing data length:\n"\
    f"X_train: {len(X_train)} | X_test: {len(X_test)} | y_train: {len(y_train)} | y_test: {len(y_test)}"
)
print(f"X_train type is {type(X_train)}")

"""
2. Building a model
Build a modle to classify the "blue" and "red" dots
To do so, we want to:
1. setup device agnostic code so our code will run on GPU if there is one
2. Construst a model (by subclassing nn.Module)
3. Define a loss function and optimizer
4. Create a training and test loop
"""
print("\n")

# Make device agnostic code
device = get_best_device()
print_device_info(device=device)

# To create a model:
# 1. Subclass nn.Module (almos alll models in PyTorch subclass from it)
# 2. Create 2 nn.Linear layers that capable of handling the shapes of our data
# 3. Define a forward() method that outline sht forward pass or forward computation of the model
# 4. Instatiat an instance of our model model class and send to target device

# 1. Define a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(
            in_features=2, # features in X_train
            out_features=5, # hidden layer, upscale to 5, a hyperparameter
        )
        self.layer_2 = nn.Linear(
            in_features=5, # hidden layer, take the 5 features from previous layer
            out_features=1, # feature in y
        )

    # 3. Define a forward() method that outline sht forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2

# 4. Instatiat an instance of our model model class and send to target device
circle_model_v0 = CircleModelV0().to(device=device)
print(f"Print out circle_model_v0: {circle_model_v0}")
print(f"circle_model_v0 is on {next(circle_model_v0.parameters()).device}")

# Note the above mode will not work because it only has linear layers, 
# which cannot handle the nonlinear classification (just draw a straight line to classify)
# refer to tensorflow playground

# Let's replicate the model above using nn.Sequential()
circle_model_v0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device=device)
print(f"Print out circle_model_v0 after using nn.Sequential: {circle_model_v0}")

# Improve the model
class CircleModuleV0Improve(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1),
        )
    
    def forward(self, x):
        return self.model(x)

circle_model_v0_improve = CircleModuleV0Improve().to(device=device)
print(f"\nPrint out circle_model_v0_improve: {circle_model_v0_improve}")
print(f"circle_model_v0_improve is on {next(circle_model_v0_improve.parameters()).device}")
print(circle_model_v0_improve.state_dict())

# Make predictions using untrained model
with torch.inference_mode():
    untrained_predictions:torch.Tensor = circle_model_v0_improve(X_test.to(device))
print(f"Lenght of predictions: {len(untrained_predictions)} | Shape: {untrained_predictions.shape}")
print(f"Lenght of test samples: {len(X_test)} | Shape {X_test.shape}")
print(f"\n First 10 predictions: \n {torch.round(untrained_predictions[:10])}") # round prediction, more near 0 will be 0, or 1
print(f"\nFirst 10 labels: {y_test[:10]}")

"""
2.1 Setup loss function and optimizer
For regression, we may use MAE or MSE
For classifiction we may use binary cross entropy or categorical corss entropy
For optimizers, two of the most common and useful are SGD and Adam

For this problem we are going to use nn.BCEWithLogitsLoss()
"""
# setup loss function and optimizer
# nn.BCELoss() requires the inputs to have gone throught the sigmoid activation function prior to input to BCEloss
loss_fn_bcelogits = nn.BCEWithLogitsLoss() # with sigmoid activation built-in

optimizer_sdg = torch.optim.SGD(
    params=circle_model_v0_improve.parameters(),
    lr=0.1
)

# Calculate accuracy - out of 100 examples, what percentage does out model get right
def accruacty_fn(y_true: torch.Tensor, y_predict: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_predict).sum().item()
    accruacy = (correct/len(y_predict)) *100
    return accruacy
