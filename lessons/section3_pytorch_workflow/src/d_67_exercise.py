import torch
from torch import nn
from pathlib import Path

from common.device import get_best_device, print_device_info
from lessons.section3_pytorch_workflow.common import plot_prediction

# first get the best device
device = get_best_device()
print_device_info(device=device)

"""
1. Create a straight line dataset using the linear regression formula (weight * X + bias).
- Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
- Split the data into 80% training, 20% testing.
- Plot the training and testing data so it becomes visual.
"""
weight = 0.3
bias = 0.9

# original data
STEP: float = 1 / 100
START: int = 0
END: int = 1
X = torch.arange(
    start=START, end=END, step=STEP, device=device, dtype=torch.float32
).unsqueeze(
    dim=1
)  # make it a column vector (n*1 matrix)
y = weight * X + bias

print(f"length of X is {len(X)}")

# split data into training and test sets
TRAIN_RATIO: float = 0.8
train_split = int(TRAIN_RATIO * len(X))
X_train, y_train = X[:train_split, :], y[:train_split, :]
X_test, y_test = X[train_split:, :], y[train_split:, :]
print(f"length of train data is {len(X_train)}")
print(f"length of test data is {len(X_test)}")

plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    fig_save_path="lessons/section3_pytorch_workflow/src/d_original_data_line_49.png",
    title="d - plot original data",
)

"""
2. Build a PyTorch model by subclassing nn.Module.
- Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for weights and one for bias.
- Implement the forward() method to compute the linear regression function you used to create the dataset in 1.
- Once you've constructed the model, make an instance of it and check its state_dict().
- Note: If you'd like to use nn.Linear() instead of nn.Parameter() you can.
"""
print("\n")


class LinearRegressionModelExercise(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(
            in_features=1,
            out_features=1,
            device=device,
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


linear_regression_model_exercise = LinearRegressionModelExercise()
print(
    f"The created linear regression model parameters are {linear_regression_model_exercise.state_dict()}"
)
print(
    f"The created linear regression model is on {next(linear_regression_model_exercise.parameters()).device}"
)

"""
3. Create a loss function and optimizer using nn.L1Loss() and torch.optim.SGD(params, lr) respectively.
- Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
- Write a training loop to perform the appropriate training steps for 300 epochs.
- The training loop should test the model on the test dataset every 20 epochs.
"""
LEARNING_RATE = 0.01

# loss function
loss_function_l1 = nn.L1Loss()

# optimizer
optimizer_sdg = torch.optim.SGD(
    params=linear_regression_model_exercise.parameters(),
    lr=LEARNING_RATE,
)

EPOCHS = 400
TEST_RATE = 20  # test data every TEST_RATE epochs

for epoch in range(EPOCHS):
    # 1. set the model to training
    linear_regression_model_exercise.train()

    # 2. zero grad the optimizer
    optimizer_sdg.zero_grad()

    # 3. forward pass
    y_predict_training: torch.Tensor = linear_regression_model_exercise(X_train)

    # 4. calculate the loss
    training_loss: torch.Tensor = loss_function_l1(y_predict_training, y_train)

    # 5. backward the loss, calculate the gradient, backpropagation
    training_loss.backward()

    # 6. step the optimizer, perform gradient decendent
    optimizer_sdg.step()

    # test the model
    if epoch % TEST_RATE == 0:
        # 1. test the model to evaluation
        linear_regression_model_exercise.eval()
        with torch.inference_mode():
            # 2. forward pass
            y_prediction_testing = linear_regression_model_exercise(X_test)

            # calculate the loss
            test_loss = loss_function_l1(y_prediction_testing, y_test)

            print(
                f"At epoch {epoch} | Training loss {training_loss} | Test loss {test_loss}"
            )

print("\n")
print(
    f"After training, the model parameters are {linear_regression_model_exercise.state_dict()}"
)
print(f"target data, weight = {weight}, bias = {bias}")

"""
4. Make predictions with the trained model on the test data.
Visualize these predictions against the original training and testing data 
(note: you may need to make sure the predictions are not on the GPU if you want 
to use non-CUDA-enabled libraries such as matplotlib to plot).
"""

