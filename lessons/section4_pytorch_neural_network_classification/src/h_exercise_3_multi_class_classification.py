import torch
from torch import nn
import torchmetrics
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common.device import get_best_device, print_device_info
from common.helper_fucntion import plot_decision_boundary

"""
7. Create a multi-class dataset using the spirals data creation function from CS231n (see below for the code).

7.1. Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).

7.2 Build a loss function and optimizer capable of handling multi-class data 
(optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).

7.3 Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy 
(you can use any accuracy measuring function here that you like).

7.4 Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary() function should work for this dataset too.
"""
# Get best device
device = get_best_device()
print_device_info(device)

# Create data set
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype="uint8")  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
# lets visualize the data
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="RdYlBu")
plt.grid()
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/h_line_36_cs231_spiral_dataset_original.png"
)

# Constant
IN_FEATURES = 2
OUT_FEATURES = K
HIDDEN_UNITS = 16
LEARNING_RATE = 0.1
LEARNING_RATE_sgd = 0.1
EPOCHS = 100
EPOCHS_sgd = 200

# Convert data to tensor and to the target device
X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.long).to(device)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train: torch.Tensor
X_test: torch.Tensor
y_train: torch.Tensor
y_test: torch.Tensor


# Construct a model and move to target device
class SpiralDataModel(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_units: int = 8
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


base_model = SpiralDataModel(
    in_features=IN_FEATURES,
    out_features=OUT_FEATURES,
    hidden_units=HIDDEN_UNITS
)
base_model = base_model.to(device)

# Make sure have an apple to apple comparsion to SGD model, have the same init state
spiral_model = copy.deepcopy(base_model)
spiral_model = spiral_model.to(device)

# loss function and optimizer
loss_fn_crossentropy = nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(
    params=spiral_model.parameters(),
    lr=LEARNING_RATE
)

# Metric functions
accuracy_calculator = torchmetrics.Accuracy(task="multiclass", num_classes=K).to(device)

# Training and Testing loop
# Set the training Constant
epochs: int = EPOCHS
epoch_95_accuracy: int = 0
epoch_95_accuracy_set_flag: bool = False

for epoch in range(epochs):
    spiral_model.train()

    # Forward pass
    y_logits_train = spiral_model(X_train)

    # Loss and accuracy
    loss_train: torch.Tensor = loss_fn_crossentropy(y_logits_train, y_train)
    accuracy_calculator.reset() # Need to reset accuracy_calculator because it is stateful
    accuracy_train = accuracy_calculator(y_logits_train, y_train)
    if accuracy_train.item() > 0.95 and not epoch_95_accuracy_set_flag:
        epoch_95_accuracy_set_flag = True
        epoch_95_accuracy = epoch
    
    # Optimizer zero grad
    optimizer_adam.zero_grad()

    # Loss backward
    loss_train.backward()

    # Optimizer Step
    optimizer_adam.step()

    # test
    spiral_model.eval()
    with torch.inference_mode():
        # Forward pass
        y_logits_test = spiral_model(X_test)

        # loss and accuracy
        loss_test: torch.Tensor = loss_fn_crossentropy(y_logits_test, y_test)
        accuracy_calculator.reset()
        accuracy_test = accuracy_calculator(y_logits_test, y_test)
    
    # Print every 100 epochs
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Training loss: {loss_train:.5f}, Training accuracy: {accuracy_train:.2f} | "
            f"Test loss: {loss_test:.5f}, Tesing accurady: {accuracy_test:.2f}"
        )

print(f"At epoch {epoch_95_accuracy}, ADAM model reaches 95% accuracy")

#=============================================================================
# Using sgd optimizer
spiral_model_sgd = copy.deepcopy(base_model)
spiral_model_sgd = spiral_model_sgd.to(device)
optimizer_sgd = torch.optim.SGD(
    params=spiral_model_sgd.parameters(),
    lr=LEARNING_RATE_sgd,  # Learning rate
    momentum=0.9  # Momentum helps SGD converge better on complex problems
)

# Create fresh metric objects for SGD training (to avoid state accumulation from Adam training)
accuracy_calculator_sgd = torchmetrics.Accuracy(task="multiclass", num_classes=K).to(device)

epoch_95_accuracy_sgd: int = 0
epoch_95_accuracy_set_flag_sgd: bool = False
for epoch in range(EPOCHS_sgd):
    spiral_model_sgd.train()

    # Forward pass
    y_logits_train_sgd = spiral_model_sgd(X_train)

    # Loss and accuracy
    loss_train_sgd: torch.Tensor = loss_fn_crossentropy(y_logits_train_sgd, y_train)
    accuracy_calculator_sgd.reset()
    accuracy_train_sgd = accuracy_calculator_sgd(y_logits_train_sgd, y_train)
    if accuracy_train_sgd.item() > 0.95 and not epoch_95_accuracy_set_flag_sgd:
        epoch_95_accuracy_set_flag_sgd = True
        epoch_95_accuracy_sgd = epoch
    
    # Optimizer zero grad
    optimizer_sgd.zero_grad()

    # Loss backward
    loss_train_sgd.backward()

    # Optimizer Step
    optimizer_sgd.step()

    # test
    spiral_model_sgd.eval()
    with torch.inference_mode():
        # Forward pass
        y_logits_test_sgd = spiral_model_sgd(X_test)

        # loss and accuracy
        loss_test_sgd: torch.Tensor = loss_fn_crossentropy(y_logits_test_sgd, y_test)
        accuracy_calculator_sgd.reset()
        accuracy_test_sgd = accuracy_calculator_sgd(y_logits_test_sgd, y_test)
    
    # Print every 100 epochs
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Training loss: {loss_train_sgd:.5f}, Training accuracy: {accuracy_train_sgd:.2f} | "
            f"Test loss: {loss_test_sgd:.5f}, Tesing accurady: {accuracy_test_sgd:.2f}"
        )

print(f"At epoch {epoch_95_accuracy_sgd}, sgd model reaches 95% accuracy")

# Visualize and Get Metrics of the model
# Visualize first
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.title("Adam optimizer Training Data")
plot_decision_boundary(model=spiral_model, X=X_train, y=y_train) # this will move model to cpu
plt.grid()
plt.subplot(2, 2, 2)
plt.title("Adam optimizer Testing Data")
plot_decision_boundary(model=spiral_model, X=X_test, y=y_test)
plt.grid()
plt.subplot(2, 2, 3)
plt.title("SGD optimizer Training Data")
plot_decision_boundary(model=spiral_model_sgd, X=X_train, y=y_train)
plt.grid()
plt.subplot(2, 2, 4)
plt.title("SGD optimizer Testing Data")
plot_decision_boundary(model=spiral_model_sgd, X=X_test, y=y_test)
plt.grid()
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/h_line_243_multiclass_prediction_boundary_adam_and_sgd.png"
)

# Deine a metrics function:
def evaluate_multiclass_model(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> dict:
    """
    Evaluate a multiclass classification model on a dataset.

    Returns a dictionary containing:
        - accuracy
        - precision
        - recall
        - f1
        - confusion_matrix
    """
    # make sure model is moved back to target device
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    model.eval()

    # Fresh metric objects so states do not mix
    accuracy_metric = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=num_classes,
    ).to(device)

    precision_metric = torchmetrics.Precision(
        task="multiclass",
        num_classes=num_classes,
        average="macro",
    ).to(device)

    recall_metric = torchmetrics.Recall(
        task="multiclass",
        num_classes=num_classes,
        average="macro",
    ).to(device)

    f1_metric = torchmetrics.F1Score(
        task="multiclass",
        num_classes=num_classes,
        average="macro",
    ).to(device)

    confusion_metric = torchmetrics.ConfusionMatrix(
        task="multiclass",
        num_classes=num_classes,
    ).to(device)

    with torch.inference_mode():
        logits = model(X)

        accuracy = accuracy_metric(logits, y).item()
        precision = precision_metric(logits, y).item()
        recall = recall_metric(logits, y).item()
        f1 = f1_metric(logits, y).item()
        confusion_matrix = confusion_metric(logits, y)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix,
    }

# Calculate the metrics
adam_results = evaluate_multiclass_model(
    model=spiral_model,
    X=X_test,
    y=y_test,
    num_classes=K,
    device=device,
)

sgd_results = evaluate_multiclass_model(
    model=spiral_model_sgd,
    X=X_test,
    y=y_test,
    num_classes=K,
    device=device,
)

# Print results
print("=== Adam Results ===")
print(f"Accuracy : {adam_results['accuracy']:.4f}")
print(f"Precision: {adam_results['precision']:.4f}")
print(f"Recall   : {adam_results['recall']:.4f}")
print(f"F1 Score : {adam_results['f1']:.4f}")
print("Confusion Matrix:")
print(adam_results["confusion_matrix"])

print("\n=== SGD Results ===")
print(f"Accuracy : {sgd_results['accuracy']:.4f}")
print(f"Precision: {sgd_results['precision']:.4f}")
print(f"Recall   : {sgd_results['recall']:.4f}")
print(f"F1 Score : {sgd_results['f1']:.4f}")
print("Confusion Matrix:")
print(sgd_results["confusion_matrix"])

"""
Sample out out for metrics:

=== Adam Results ===
Accuracy : 1.0000
Precision: 1.0000
Recall   : 1.0000
F1 Score : 1.0000
Confusion Matrix:
tensor([[22,  0,  0],
        [ 0, 16,  0],
        [ 0,  0, 22]], device='cuda:0')

=== SGD Results ===
Accuracy : 0.9500
Precision: 0.9463
Recall   : 0.9489
F1 Score : 0.9464
Confusion Matrix:
tensor([[20,  2,  0],
        [ 0, 15,  1],
        [ 0,  0, 22]], device='cuda:0')
"""
