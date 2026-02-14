import torch
from torch import nn

# import matplotlib
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from pathlib import Path

# from torchviz import make_dot

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
Visualize data -- helpful to understand the data better
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


# define a function
def plot_prediction(
    train_data: torch.Tensor = X_train,
    train_labels: torch.Tensor = y_train,
    test_data: torch.Tensor = X_test,
    test_labels: torch.Tensor = y_test,
    predictions: torch.Tensor | None = None,
    fig_save_path: str = "lessons/section3_pytorch_workflow/src/b_train_test_split_plot_in_function.png",
):
    """
    Function to plot the above training and test data/label

    :param train_data: training data to plot
    :param train_labels: labels to plot
    :param test_data: test data to plot
    :param test_labels: test labels to plot
    :param predictions: predictions to plot, if any
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(
        train_data.squeeze().numpy(),  # best practice to squeeze because matplotlib need 1D numpy array
        train_labels.squeeze().numpy(),
        color="blue",
        label="Training data",
    )
    # Test data
    plt.scatter(
        test_data.squeeze().numpy(),
        test_labels.squeeze().numpy(),
        color="green",
        label="Test data",
    )

    # prediction
    if predictions is not None:
        plt.scatter(
            test_data.squeeze().numpy(),
            predictions.squeeze().numpy(),
            color="red",
            label="Predictions",
        )

    plt.legend()
    plt.grid(True)

    plt.savefig(fig_save_path)  # best practice is to store the figure


# call the plot function
plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    fig_save_path="lessons/section3_pytorch_workflow/src/b_train_test_split_plot_in_function.png",
)


"""
Training process:
1. Start with random number
2. Look at training data and adjust the random values to better represent (or get closer to the ideal values)
Two algorithms: 1. Gradient descent; 2. Backpropagation
"""


# Building the first PyTorch model, linear regression model class
# need to import nn first
class LinearRegressionModel(
    nn.Module
):  # <- almost everything in PyTorch inherhits from nn.Module
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(  # nn.Parameter can be hard-coded value
            torch.randn(
                size=[1],
                requires_grad=True,
                dtype=torch.float32,
            )
        )
        self.bias = nn.Parameter(
            torch.randn(
                size=[1],
                requires_grad=True,
                dtype=torch.float32,
            )
        )

    # Forward method to defind the computation in the model
    # This need to be overrided, this is what the module does
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


"""
Pytorch model building essentials
1. torch.nn --  contains all the building blocks for computational graphs (a neural network can be considered as a computational graph)
2. torch.nn.Parameter -- what parameter our model try and learn, often a PyTroch layer from torch.nn will set these for us
3. torch.nn.Module -- base class for all neural network module, if you subclass it, you should override the forward() method
4. torch.optim -- where the optimizers in Pytorch live, they will help with gradient descent
5. def forward() -- All nn.Module subclasses require you to override it, this method defines what happens in the forward computation
"""

print("=============================================================")
print("\nstart doing something for the created torch.nn.Module")
# checking content of PyTorch model "LinearRegressionModel"
# There is a method ".parameters()"

# creating a random seed -- make sure to reproduce the value for learning
torch.manual_seed(42)

# instance of "LinearRegressionModel"
linear_regression_model = LinearRegressionModel()

# check out the parameters
linear_regression_model_parameters = list(linear_regression_model.parameters())
print(
    f"The parameter in the linear_regression_model is \n {linear_regression_model_parameters}"
)

# List name parameters
linear_regression_model_parameters_listed = linear_regression_model.state_dict()
print(
    f"The listed name parameter of linear_regression_model are {linear_regression_model_parameters_listed}"
)

# making prediction using torch.inference_mode(), predict "y_test" using "X_test"
# when data go through model, it will go through the "forward()" method
with torch.inference_mode():
    # torch.inference_mode tells PyTorch only to the math calculation in forward(), do no remember everything (change the value of tensor, etc.)
    # also the inference/prediction will be much faster
    y_predictions_no_training = linear_regression_model(X_test)

# visulize the y_prediction to compare with the actual data
plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_predictions_no_training,
    fig_save_path="lessons/section3_pytorch_workflow/src/b_compare_prediction_with_actual_no_train_model_code_line_220.png",
)

"""
* Training model:
"Training" is to make model to move from some "random" parameters to some "known" parameters
Or from a poor representation fo data to a better representation of data
One way to measure how "poor/wrong" the model predictions are, using loss function

Note: loss function == cost function == criterion

Things we need to train the model:
1. Loss function: A function to measure how wrong the model's prediction compare to the ideal output, the lower the better
2. Optimizer: Takes into account the loss of a model and adjust the model's parameters to improve the loss function

Specifically for Pytorch, we need:
* A training loop
* A testing loop
"""

# In this case, we will use nn.L1Loss, which is MAE
loss_fn_mae = nn.L1Loss()

# Setup an optimizer (stochastic gradient decent)
# In optimizer, we will often need to set two parameter
# params - the model parameters you'd like to optimize, needs to be nn.Parameter objects
# lr - learning rate, a hyperparameter that defines how big/small the optimizer changes the parameters with each step
# Parameter vs hyperparameter: parameters are part of the model, will be updated during trainig; hyperparameter are set before training by software engineer/data scientist
optimizer_sgd = torch.optim.SGD(
    params=linear_regression_model.parameters(),
    lr=0.01,  # lr = learning rate, possibly the most important hyperparameter you can set
)

"""
Building a trainig loop (and a testing loop) in PyTorch
Need in training loop:
0. Loop through the data and do...
1. forward pass (involoves data moving trhough our model's forward() function) to make predictions on data -- also called forward propagation
2. Calculate the loss (compare forward pass predictions to ground truth labels)
3. Optimizer zero grad
4. Loss backward -- move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loos (backpropagation)
5. Optimizer step -- use the optimizer to adjust our models's parameters to try and improve the loss (gradient descent)
"""
# set a random seed again for learning/reproducible
# torch.manual_seed(42)
print("=============================================================")
# And epoch is one loop through the data (a hyperparameter, we set by ourselves)
epochs = 200

# List name parameters again before training
print(
    f"\nlinear_regression_model parameters before training are {linear_regression_model_parameters_listed}"
)

print(f"target values are: weight = {weight}, bias = {bias}")

# Track different values, for comparsion with other training approaches of the model (different hyperparameters)
epoch_count: list[int] = []
loss_values: list[float] = []
test_loss_values: list[float] = []

### Training, the following code can be written in a function
# 0. loop through the data
for epochs in range(epochs):
    # set the model to training mode
    linear_regression_model.train()  # train mode set Sets an internal flag: model.training = True

    optimizer_sgd.zero_grad()  # zero grad better before forward pass
    # 1. Forward pass
    y_train_prediction = linear_regression_model(X_train)

    # 2. Calculate the loss
    loss: torch.Tensor = loss_fn_mae(
        y_train_prediction, y_train
    )  # always (prediction, targets) <-- follow this form in loss function
    # print(f"in loop {epochs}, loss = {loss}")

    # 3. Optimizer zero grad
    # optimizer_sgd.zero_grad()

    # 4. perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient decent)
    optimizer_sgd.step()  # by default how the optimizer changes will accumulate through the loop, we will need to zero them above at step 3

    # Make graph
    # dot = make_dot(loss, params=dict(linear_regression_model.named_parameters()))
    # dot.format = "png"
    # dot.render("lessons/section3_pytorch_workflow/src/b_graph_linear_regression", cleanup=True)

    ### Testing
    linear_regression_model.eval()  # turn off different settings in the model not needed for evaluation/testing (dropout/BatchNorml layers)
    # print(f"in loop {epochs}, model parameter = {linear_regression_model_parameters_listed}")
    with torch.inference_mode():  # turuns off gradient tracking and a couple of more things behind the scenes
        # 1. Do the forward pass
        test_prediction = linear_regression_model(X_test)

        # 2. calculate the loss
        test_loss: torch.Tensor = loss_fn_mae(
            test_prediction, y_test
        )  # (prediction, targets)

    # print out what is happening during testing
    if epochs % 10 == 0:
        epoch_count.append(epochs)
        loss_values.append(
            loss.item()
        )  # item convert a tensor has a single value to Python value (use to_list() for tensors with more than one element)
        test_loss_values.append(test_loss.item())
        print(f"Epoch: {epochs} | Training Loss: {loss} | Test loss: {test_loss}")
        print(f"model parameters {linear_regression_model.state_dict()}")

print(
    f"After training loop, linear regression model parameters are {linear_regression_model_parameters_listed}"
)

# plot loss curves
plt.figure(figsize=(10, 7))
plt.plot(epoch_count, loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss (MAE)")
plt.legend()
plt.grid(True)
plt.savefig(
    "lessons/section3_pytorch_workflow/src/b_train_loss_and_epochs_plot_code_line_352.png"
)


with torch.inference_mode():
    y_prediction_after_training = linear_regression_model(X_test)

# plot after training loop:
plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_prediction_after_training,
    fig_save_path="lessons/section3_pytorch_workflow/src/b_compare_prediction_with_actual_after_training_loop_code_line_361.png",
)

"""
Saving a model in PyTorch
There are 3 main methods to save and load models in PyTorch:
1. torch.save() - allows you to save a PyTorch object in Python's pickle format
2. torch.load() - allows you load a saved PyTorch objectg
3. torch.nn.Module.load_state_dict() - allows to load a model's saved state dictionary
"""

# Saving the PyTorch model

# 1. Create models directory
MODEL_PATH = Path("lessons/section3_pytorch_workflow/src/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create a model save path
MODEL_NAME = "section3_b_linear_regression_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state_dict
print(f"\n Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=linear_regression_model.state_dict(), f=MODEL_SAVE_PATH)

# Loading a PyTorch model
# Since we saved out model's state_dict() rather than entire moedel, we will create a new instance of out model class and load the saved state_dict()
# To load in a saved state_dict(), we have to instantiate a new instance of our model
loaded_linear_regression_model = LinearRegressionModel()

# Load the saved state_dict of linear_regression_model (this will update the new instance with updated parameters)
loaded_linear_regression_model.load_state_dict(
    torch.load(f=MODEL_SAVE_PATH, weights_only=True)
)

print(f"Original model parameters {linear_regression_model_parameters_listed}")
print(f"loaded model parameters are {loaded_linear_regression_model.state_dict()}")

# make some predictions using the loaded model
loaded_linear_regression_model.eval()  # remember this eval() first, then inference_mode, best practice
with torch.inference_mode():
    y_prediction_loaded = loaded_linear_regression_model(X_test)

plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_prediction_loaded,
    fig_save_path="lessons/section3_pytorch_workflow/src/b_compare_prediction_with_actual_after_loading_model_code_line_412.png",
)
