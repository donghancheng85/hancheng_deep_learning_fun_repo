import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from common.helper_fucntion import plot_decision_boundary
from lessons.section3_pytorch_workflow.common import plot_prediction

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
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/b_visualize_generated_circle_data.png"
)

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
print(
    f"Shape of one sample of X: {X_sample.shape}, y: {y_sample.shape}"
)  # two features for input X, one feature for y

"""
1.2 Turn data into tensors, create train and test splits
"""
# Turn data into tensors
X = torch.from_numpy(X).type(torch.float32)  # ndarray default dtype is float64
y = (
    torch.from_numpy(y).type(torch.float32).unsqueeze(dim=1)
)  # need to unsqueeze because the model output shape will be (feature_num, other_dim)

print(
    f"After convert to Tensor, X first 5 elements:\n {X[:5]}, y first 5 elements:\n {y[:5]}"
)

# Split into training and test dataset, using randmo approach

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)  # 20% test size, 80% train
X_train: torch.Tensor
X_test: torch.Tensor
y_train: torch.Tensor
y_test: torch.Tensor

print(
    "Training and testing data length:\n"
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
            in_features=2,  # features in X_train
            out_features=5,  # hidden layer, upscale to 5, a hyperparameter
        )
        self.layer_2 = nn.Linear(
            in_features=5,  # hidden layer, take the 5 features from previous layer
            out_features=1,  # feature in y
        )

    # 3. Define a forward() method that outline sht forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x))  # x -> layer_1 -> layer_2


# 4. Instatiat an instance of our model model class and send to target device
circle_model_v0 = CircleModelV0().to(device=device)
print(f"Print out circle_model_v0: {circle_model_v0}")
print(f"circle_model_v0 is on {next(circle_model_v0.parameters()).device}")

# Note the above mode will not work because it only has linear layers,
# which cannot handle the nonlinear classification (just draw a straight line to classify)
# refer to tensorflow playground

# Let's replicate the model above using nn.Sequential()
circle_model_v0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5), nn.Linear(in_features=5, out_features=1)
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
print(
    f"circle_model_v0_improve is on {next(circle_model_v0_improve.parameters()).device}"
)
print(circle_model_v0_improve.state_dict())

# Make predictions using untrained model
with torch.inference_mode():
    untrained_predictions: torch.Tensor = circle_model_v0_improve(X_test.to(device))
print(
    f"Lenght of predictions: {len(untrained_predictions)} | Shape: {untrained_predictions.shape}"
)
print(f"Lenght of test samples: {len(X_test)} | Shape {X_test.shape}")
print(
    f"\n First 10 predictions: \n {torch.round(untrained_predictions[:10])}"
)  # round prediction, more near 0 will be 0, or 1
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
loss_fn_bcelogits = nn.BCEWithLogitsLoss()  # with sigmoid activation built-in
print(f"type of loss_fn_bcelogits is {type(loss_fn_bcelogits)}")
optimizer_sdg = torch.optim.SGD(params=circle_model_v0_improve.parameters(), lr=0.1)


# Calculate accuracy - out of 100 examples, what percentage does out model get right
def accruacty_fn(y_true: torch.Tensor, y_predict: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_predict).sum().item()
    accruacy = (correct / len(y_predict)) * 100
    return accruacy


"""
3. Train model
To train model, we need a training loop with the following steps
1. Forward pass
2. calculate the loss
3. optimize zero grad
4. loss backward backpropagation
5. Optimizer step (gradient descent)
"""

"""
3.1 Ging from raw logits -> prediction probabilities -> prediction labels
our model outputs are going to be raw logits
We can convert the "logits" into prediction probabilities by passing them to some kind of
activation function (e.g., sigmoid for binary classification and softmax for multiclass classification)
Then we can convert out model predictions to prediction labels by either rounding them (binary classification)
or taking the max (multiclass classification)
"""

# View the first 5 outputs of the forward pass on the test data
circle_model_v0_improve.eval()  # need to remember to use eval and inference_mode when making predictions
with torch.inference_mode():
    y_logits_first5 = circle_model_v0_improve(X_test.to(device=device))[
        :5
    ]  # need to be on the same device for the model and input
print(f"\nFirst 5 element of first f outputs\n{y_logits_first5}")

# Using sigmoid activation function on our model logits into prediction probabilities
y_prediction_probabilities_first5 = torch.sigmoid(y_logits_first5)
print(
    f"the probabilities after converting the logits are\n{y_prediction_probabilities_first5}"
)

# for the prediction probability values, we need t operfomr a range-style rounding on them >=0.5 ->1

# Find the predicted labels
y_prediction_after_round_first5 = torch.round(y_prediction_probabilities_first5)
print(f"After round, the prediction outputs are\n {y_prediction_after_round_first5}")

# In full, we should use
circle_model_v0_improve.eval()
with torch.inference_mode():
    # logits -> predction probabilities -> prediction labels
    y_prediction_labels = torch.round(
        torch.sigmoid(circle_model_v0_improve(X_test.to(device=device)))
    )[:5]

# Check for equality
print(
    torch.eq(y_prediction_after_round_first5, y_prediction_labels)
)  # check if they are the same

"""
3.2 building training and testing loop
"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # just for reproducibility

epochs = 100
# put data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and eval loop
for epoch in range(epochs):
    # Training
    circle_model_v0_improve.train()

    # 1. forward pass
    y_logits = circle_model_v0_improve(X_train)
    y_prediction = torch.round(
        torch.sigmoid(y_logits)
    )  # turn logits -> prediction probabilities -> prediction labels

    # 2. calculate loss and accuracy
    loss_train: torch.Tensor = loss_fn_bcelogits(
        y_logits, y_train
    )  # nn.BCEwithLogitsLoss expetes raw logits as input
    accuracy_train = accruacty_fn(y_true=y_train, y_predict=y_prediction)

    # 3 Optimzer zero grad
    optimizer_sdg.zero_grad()

    # 4. loss backward (backpropagation)
    loss_train.backward()

    # 5. optimizer step (gradient descent)
    optimizer_sdg.step()

    # Testing
    circle_model_v0_improve.eval()
    with torch.inference_mode():
        # 1. forward pass
        y_logits_test = circle_model_v0_improve(X_test)
        y_test_prediction = torch.round(torch.sigmoid(y_logits_test))

        # 2. calculate test loss/accuracy
        loss_test = loss_fn_bcelogits(y_logits_test, y_test)
        accuracy_test = accruacty_fn(y_true=y_test, y_predict=y_test_prediction)

        # print out what's happening
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Training loss: {loss_train:.5f}, Training accuracy: {accuracy_train:.2f}% | "
                f"Test loss: {loss_test:.5f}, Tesing accurady: {accuracy_test:.2f}%"
            )


# above training results will be bad

"""
4. Make predictions and evaluate the model
The model is not "learning". To inspect, we are going to make some predictions and make them visual
To do so, we are going to import a function called "plot_decision_boundary()"
"""
# Download helper functions from Learn PyTorch repo
if Path("common/helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    pass
    # print("Download helper_function.py")
    # request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    # with open("common/helper_fucntion.py", "wb") as file:
    #     file.write(request.content)

# plot decision boundary of model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(circle_model_v0_improve, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(circle_model_v0_improve, X_test, y_test)
# plt.savefig("lessons/section4_pytorch_neural_network_classification/src/b_compare_linear_layer_train_and_test.png")


"""
5. improve the model (from a model perspective)
* add more layers - give model more chances to learn about patterns in data
* add more hidden units - go from 5 hidden units to 10 hidden units (for example in this code)
* Fit for longer (more epochs)
* Changing the activation function (add the activation function into other layers other than output layer)
* Change the learning rate
* Change the loss functions

These options are all from a model's perspective because they deal directly with the model, rather than the data.
Because the options are all values we can change, they are refer to as "hyperparameters"
"""

print("==========================================")


# try improve the model by
# - adding hidden unit 5 -> 10
# - increase the number of layers 2 -> 3
# - increase number of epochs 100 -> 1000
class CircleModuleV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))


circle_model_v1 = CircleModuleV1().to(device=device)

# create a loss function for this
loss_fn_bcelogits_v1 = nn.BCEWithLogitsLoss()

# optimizer
optimizer_sdg_v1 = torch.optim.SGD(params=circle_model_v1.parameters(), lr=0.1)

# Manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

epochs = 1000

# Put data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training loop
for epoch in range(epochs):
    # Training
    circle_model_v1.train()

    y_logits_v1 = circle_model_v1(X_train)
    y_prediction_v1 = torch.round(torch.sigmoid(y_logits_v1))

    loss_train_v1: torch.Tensor = loss_fn_bcelogits_v1(y_logits_v1, y_train)
    accuracy_train_v1 = accruacty_fn(y_true=y_train, y_predict=y_prediction_v1)

    optimizer_sdg_v1.zero_grad()

    loss_train_v1.backward()

    optimizer_sdg_v1.step()

    # Testing
    circle_model_v1.eval()
    with torch.inference_mode():
        y_logits_test_v1 = circle_model_v1(X_test)
        y_prediction_test_v1 = torch.round(torch.sigmoid(y_logits_test_v1))

        loss_test_v1 = loss_fn_bcelogits_v1(y_logits_test_v1, y_test)
        accuracy_test_v1 = accruacty_fn(y_true=y_test, y_predict=y_prediction_test_v1)

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Training loss: {loss_train_v1:.5f}, Training accuracy: {accuracy_train_v1:.2f}% | "
                f"Test loss: {loss_test_v1:.5f}, Tesing accurady: {accuracy_test_v1:.2f}%"
            )

# Plot decision boundary again
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(circle_model_v1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(circle_model_v1, X_test, y_test)
# plt.savefig("lessons/section4_pytorch_neural_network_classification/src/b_add_more_linear_layer_and_check_prediction.png")

"""
5.1 Preparing data to see the model can fit a straight line
One way to troubleshoot a larger problem is to test a smaller problem
"""
print("======================================================\n")
# Create some data (straight line, a smaller problem to test if the model can learn something)
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)  # a matrix
y_regression = weight * X_regression + bias

# split train/test
train_split = int(0.8 * len(X_regression))  # 80% of X will be training
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]
print(f"len of X_train_regression {len(X_train_regression)} | X_test_regression {len(X_test_regression)}")

plot_prediction(
    train_data=X_train_regression,
    train_labels=y_train_regression,
    test_data=X_test_regression,
    test_labels=y_test_regression,
    fig_save_path="lessons/section4_pytorch_neural_network_classification/src/b_smaller_problem_linear_data_origin.png",
    title="plot 'smaller problem' before training",
)

"""
5.2 Adjusting circle_model_v1 to fit a straight line
"""
# Same architect as circle_model_v1 using nn.Sequential()
similar_structure_model = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1),
).to(device=device)

# Loss and optimizer
loss_fn_l1 = nn.L1Loss()
optimizer_sdg_regression = torch.optim.SGD(
    params=similar_structure_model.parameters(),
    lr=0.01,
)

# Train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Epochs
epochs = 2000

# Put the data on target
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# Training
for epoch in range(epochs):
    similar_structure_model.train()

    y_prediction_train_regression = similar_structure_model(X_train_regression)
    loss_train_regression: torch.Tensor = loss_fn_l1(y_prediction_train_regression, y_train_regression)
    optimizer_sdg_regression.zero_grad()
    loss_train_regression.backward()
    optimizer_sdg_regression.step()

    # testing
    similar_structure_model.eval()
    with torch.inference_mode():
        y_prediction_test_regression = similar_structure_model(y_test_regression)
        loss_test_regression: torch.Tensor = loss_fn_l1(y_prediction_test_regression, y_test_regression)
    
    # Printing out
    if epoch % 100 == 0:
        print(
                f"Epoch: {epoch} | "
                f"Training loss: {loss_train_regression:.5f} | "
                f"Test loss: {loss_test_regression:.5f}"
            )

# Plot the data after training
similar_structure_model.eval()
with torch.inference_mode():
    y_prediction_after_training_regression = similar_structure_model(y_test_regression)

plot_prediction(
    train_data=X_train_regression,
    train_labels=y_train_regression,
    test_data=X_test_regression,
    test_labels=y_test_regression,
    predictions=y_prediction_after_training_regression,
    fig_save_path="lessons/section4_pytorch_neural_network_classification/src/b_smaller_problem_linear_data_after_training.png",
    title="lot 'smaller problem' after training",
)
