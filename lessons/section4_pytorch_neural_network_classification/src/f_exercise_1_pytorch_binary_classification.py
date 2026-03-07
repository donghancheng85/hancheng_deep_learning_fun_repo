import torch
from torch import nn
from torchmetrics import Accuracy

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from common.device import get_best_device, print_device_info
from common.helper_fucntion import plot_decision_boundary

"""
0. Device agnostic code, constant
"""
device = get_best_device()
print_device_info(device)

NUM_CLASS = 1
NUM_FEATURE = 2
HIDDEN_UNITS = 34
LEARNING_RATE = 0.1
EPOCHS = 1000

"""
1. Make a binary classification dataset with Scikit-Learn's make_moons() function.
   For consistency, the dataset should have 1000 samples and a random_state=42.
   Turn the data into PyTorch tensors. Split the data into training and test sets using 
   train_test_split with 80% training and 20% testing.
"""

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# visualize data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu")
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/f_line_26_original_data.png"
)

# Convert data into tensors and send to target device
X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device).unsqueeze(dim=1)

# Splict train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train: torch.Tensor
X_test: torch.Tensor
y_train: torch.Tensor
y_test: torch.Tensor

"""
2. Build a model by subclassing nn.Module that incorporates 
non-linear activation functions and is capable of fitting the data you created in 1.

Feel free to use any combination of PyTorch layers (linear and non-linear) you want.
"""


class BinaryClassificationModel(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_units: int = 8
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


# Create an instance of the model and send to target device (cuda)
moon_model = BinaryClassificationModel(
    in_features=NUM_FEATURE, out_features=NUM_CLASS, hidden_units=HIDDEN_UNITS
).to(device)

"""
3. Setup a binary classification compatible loss function and optimizer to use when training the model.
"""
# loss function
loss_fn_bcewithlogits = nn.BCEWithLogitsLoss()

# Optimizer
optimizer_sdg = torch.optim.SGD(
    params=moon_model.parameters(),
    lr=LEARNING_RATE,
)

# Accuracy function
accuracy_calculator = Accuracy(task="binary").to(device)

"""
4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
- To measure model accuracy, you can create your own accuracy function or use the accuracy function in TorchMetrics.
- Train the model for long enough for it to reach over 96% accuracy.
- The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.
"""
# Set the training Constant
epochs: int = EPOCHS
epoch_96_accuracy: int = 0
epoch_96_accuracy_set_flag: bool = False


# Traing and testing loop
for epoch in range(epochs):
    # Set model to train mode
    moon_model.train()

    # forward pass
    y_logits_train = moon_model(X_train)

    # Calculate loss and accuracy
    loss_train: torch.Tensor = loss_fn_bcewithlogits(y_logits_train, y_train)
    accuracy_train = accuracy_calculator(y_logits_train, y_train)
    if accuracy_train > 0.96 and not epoch_96_accuracy_set_flag:
        # only set once when training accuracy reaches 96%
        epoch_96_accuracy = epoch
        epoch_96_accuracy_set_flag = True

    # optimizer zero grad
    optimizer_sdg.zero_grad()

    # loss backward
    loss_train.backward()

    # optimizer step
    optimizer_sdg.step()

    # Evaluation
    moon_model.eval()
    with torch.inference_mode():
        # forward pass
        y_logits_test = moon_model(X_test)

        # Calculate the loss and accuracy
        loss_test = loss_fn_bcewithlogits(y_logits_test, y_test)
        accuracy_test = accuracy_calculator(y_logits_test, y_test)

    # print the result
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Training loss: {loss_train:.5f}, Training accuracy: {accuracy_train:.2f}% | "
            f"Test loss: {loss_test:.5f}, Tesing accurady: {accuracy_test:.2f}%"
        )

print(f"At epoch {epoch_96_accuracy}, model reaches 96% accuracy")


"""
5. Make predictions with your trained model and plot them using the 
plot_decision_boundary() function created in this notebook.
"""
# Make prediction of the trained model
moon_model.eval()
with torch.inference_mode():
    y_logits_trained = moon_model(X_test)

accuracy_trained = accuracy_calculator(y_logits_trained, y_test)
print(f"After training, the accuracy is {accuracy_trained:.2f}%")

# visulize prediction
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Training Data")
plot_decision_boundary(model=moon_model, X=X_train, y=y_train)
plt.grid()
plt.subplot(1, 2, 2)
plt.title("Test Data")
plot_decision_boundary(model=moon_model, X=X_test, y=y_test)
plt.grid()
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/f_line_178_prediction_boundary.png"
)
