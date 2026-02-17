import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

from common.device import get_best_device, print_device_info
from lessons.section3_pytorch_workflow.common import plot_prediction

# check pytroch version
print(f"current Pytroch version is {torch.__version__}")

# Device agnostic code (uisng stuff in common)
device = get_best_device()
print_device_info(device)

# Create some data using the linear regression formula of y = weithg * X + bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start=start, end=end, step=step, device=device).unsqueeze(dim=1)

y = weight * X + bias

# Split the data
train_split = int(0.8 * len(X))
X_train = X[:train_split, :]
y_train = y[:train_split, :]
X_test = X[train_split:, :]
y_test = y[train_split:, :]

# plot the data
plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    fig_save_path="lessons/section3_pytorch_workflow/src/c_original_data_line_42.png",
    title="plot original data",
)


# Building a PyTorch Linear model
class LinearRegressionModulev1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                size=[1],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
        )

        self.bias = nn.Parameter(
            torch.randn(
                size=[1],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


# Create a linear by subclassing nn.Module
class LinearRegressionModelv2(nn.Module):
    """
    Using torch buildin layer nn.Linear to build the linear regression model
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        # use nn.Linear() to create a linear regression parameters
        # also called: linear transform, probing layer, fully conneected layerk, dense layer
        self.linear_layer = nn.Linear(
            in_features=1,
            out_features=1,
            device=device,
        )  # in_feature and out_feature depend on the data shape we are dealing with

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# Set the manual seed]
torch.cuda.manual_seed_all(42)
linear_regression_model_v2 = LinearRegressionModelv2()

print(
    f"state_dict of linear_regression_model_v2 is {linear_regression_model_v2.state_dict()}"
)

# Check the model current device
current_device = next(linear_regression_model_v2.parameters()).device
print(f"current device of linear_regression_model_v2 is {current_device}")

# Set the model to use the target device
# linear_regression_model_v2.to(device=device) # we already put it on GPU so it is not needed

# Training code: Loss function, Optimizer, Training Loop, Testing Loop
# Setup loss function
loss_fn_mae = nn.L1Loss()  # MAE

# Setup optimizer
optimizer_sdg = torch.optim.SGD(
    params=linear_regression_model_v2.parameters(),
    lr=0.01,
)

# Training loop
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

epochs = 450

# Put data on the target device (cuda for best performance)
# X_train = X_train.to(device=device) # no need because we already doen that in the beginning

for epoch in range(epochs):
    # 0. set model to train
    linear_regression_model_v2.train()

    # 1. zero grad
    optimizer_sdg.zero_grad()

    # 2. forward pass
    y_predict = linear_regression_model_v2(X_train)

    # 3. calculate loss
    training_loss: torch.Tensor = loss_fn_mae(y_predict, y_train)

    # 4. backpropagation, calculate gradient
    training_loss.backward()

    # 5. gradient decent
    optimizer_sdg.step()

    # Testing
    linear_regression_model_v2.eval()
    with torch.inference_mode():
        test_prediction = linear_regression_model_v2(X_test)
        test_loss = loss_fn_mae(test_prediction, y_test)

    # print out
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Training Loss: {training_loss} | Test Loss {test_loss}"
        )

print(f"after training, model parameter is {linear_regression_model_v2.state_dict()}")

# Making predictions of the model
# Trun model into evluation mode
linear_regression_model_v2.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_predict_after_training = linear_regression_model_v2(X_test)

plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_predict_after_training,
    fig_save_path="lessons/section3_pytorch_workflow/src/c_prediction_after_traning_line_170.png",
    title="plot prediction after training",
)

# Saving and loading a model

# save the model
MODEL_PATH = Path("lessons/section3_pytorch_workflow/src/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "section3_c_linear_regression_model_cuda.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"\n Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=linear_regression_model_v2.state_dict(), f=MODEL_SAVE_PATH)

# load the model
linear_regression_model_v2_loaded = LinearRegressionModelv2()
linear_regression_model_v2_loaded.load_state_dict(
    torch.load(f=MODEL_SAVE_PATH, map_location=device, weights_only=True)
)

linear_regression_model_v2_loaded.to(device=device)

print(f"linear_regression_model_v2_loaded is on {next(linear_regression_model_v2_loaded.parameters()).device}")

# use loaded model for prediction
linear_regression_model_v2_loaded.eval()
with torch.inference_mode():
    y_predict_using_loaded_model = linear_regression_model_v2_loaded(X_test)

# plot the loaded prediction
plot_prediction(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_predict_after_training,
    fig_save_path="lessons/section3_pytorch_workflow/src/c_prediction_after_loading_line_205.png",
    title="plot prediction after loading model",
)
