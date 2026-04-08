import torch
from torch import nn
from typing import Callable, Tuple
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device | str,
) -> dict[str, str | float]:
    """
    Evaluates a model on a given data_loader and returns loss and accuracy metrics.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of (X, y).
        loss_fn (torch.nn.Module): Loss function to compute per-batch loss.
        accuracy_fn (Callable[[torch.Tensor, torch.Tensor], float]): Function that takes
            (y_true, y_pred) tensors and returns an accuracy value.

    Returns:
        dict[str, str | float]: A dictionary with keys:
            - "model_name": name of the model class
            - "model_loss": average loss over the data_loader
            - "model_accuracy": average accuracy over the data_loader
    """
    loss, accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_logits_prediction: torch.Tensor = model(X)

            # Accumulate the loss and accuracy values per batch
            loss_batch = loss_fn(y_logits_prediction, y)
            loss += loss_batch
            accuracy_batch = accuracy_fn(y, y_logits_prediction.argmax(dim=1))
            accuracy += accuracy_batch

        # Scale loss and accuracy to find the average loss/acc per batch
        loss /= len(data_loader)
        accuracy /= len(data_loader)
        loss: torch.Tensor

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_accuracy": accuracy,
    }


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device | str,
) -> Tuple[float, float]:
    """
    Performs one epoch of training on the given data_loader.

    Iterates over all batches, computes forward pass, loss, and accuracy,
    then backpropagates and updates model parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training batches of (X, y).
        loss_fn (torch.nn.Module): Loss function to compute per-batch training loss.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        accuracy_fn (Callable[[torch.Tensor, torch.Tensor], float]): Function that takes
            (y_true, y_pred) tensors and returns an accuracy value.
        device (torch.device): The device to move data to before computation.
    """
    train_loss, train_accuracy = 0, 0

    # Put model into training mode and target device
    model.train()

    # A a loop to loop through the taining batches
    for batch, (X, y) in enumerate(data_loader):
        # put data on target device
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_logits_train = model(X)

        # 2. calculate the loss accruacy (per batch)
        loss_train_batch: torch.Tensor = loss_fn(y_logits_train, y)
        train_loss += (
            loss_train_batch.item()
        )  # accumulate the training loss so we can calculate the average loss of the batches
        train_accuracy += accuracy_fn(y, y_logits_train.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss_train_batch.backward()

        # 5. optimizer step
        optimizer.step()  # model parameter will be updated once per batch

    # Divide total train loss and accuracy by length of data_loader
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}%")
    return train_loss, train_accuracy


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device | str,
) -> Tuple[float, float]:
    """
    Performs one epoch of evaluation on the given data_loader.

    Runs the model in inference mode over all test batches, accumulating
    loss and accuracy, then prints the averaged results.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing test batches of (X, y).
        loss_fn (torch.nn.Module): Loss function to compute per-batch test loss.
        accuracy_fn (Callable[[torch.Tensor, torch.Tensor], float]): Function that takes
            (y_true, y_pred) tensors and returns an accuracy value.
        device (torch.device): The device to move data to before computation.
    """
    test_loss, test_accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # 1. Forward pass
            y_logits_test: torch.Tensor = model(X_test)

            # 2. Calculate the loos (accumulate)
            loss_test_batch = loss_fn(y_logits_test, y_test)
            test_loss += loss_test_batch.item()

            # 3. calculate the accuracy
            test_accuracy += accuracy_fn(y_test, y_logits_test.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(data_loader)

        # Calculate the test accuracy aversge per batch
        test_accuracy /= len(data_loader)

        print(f"\nTest loss: {test_loss:.5f}, Test accuracy: {test_accuracy:.2f}%\n")
    return test_loss, test_accuracy


def train(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict[str, list[float] | str]:
    """
    Trains and tests a PyTorch model for a number of epochs.

    Args:
        model (torch.nn.Module): The PyTorch model to train and test.
        train_data_loader (torch.utils.data.DataLoader): DataLoader providing training batches of (X, y).
        test_data_loader (torch.utils.data.DataLoader): DataLoader providing test batches of (X, y).
        loss_fn (torch.nn.Module): Loss function to compute per-batch training and test loss.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters during training.
        accuracy_fn (Callable[[torch.Tensor, torch.Tensor], float]): Function that takes
            (y_true, y_pred) tensors and returns an accuracy value.
        device (torch.device): The device to move data to before computation.
        epochs (int): Number of epochs to train and test for.

    Returns:
        dict[str, list[float] | str]: A dictionary with keys:
            - "model_name": name of the model class
            - "train_loss": list of average training loss values per epoch
            - "train_accuracy": list of average training accuracy values per epoch
            - "test_loss": list of average test loss values per epoch
            - "test_accuracy": list of average test accuracy values per epoch
    """
    # Create empty lists to store metrics across epochs
    train_loss_values = []
    train_accuracy_values = []
    test_loss_values = []
    test_accuracy_values = []

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_accuracy = train_step(
            model=model,
            data_loader=train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        test_loss, test_accuracy = test_step(
            model=model,
            data_loader=test_data_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        print(f"Epoch {epoch} metrics:")
        print(f"Train loss: {train_loss:.5f}, Train accuracy: {train_accuracy:.4f}%")
        print(f"Test loss: {test_loss:.5f}, Test accuracy: {test_accuracy:.4f}%")

        # Append metrics for this epoch
        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)
        test_loss_values.append(test_loss)
        test_accuracy_values.append(test_accuracy)

    return {
        "model_name": model.__class__.__name__,
        "train_loss": train_loss_values,
        "train_accuracy": train_accuracy_values,
        "test_loss": test_loss_values,
        "test_accuracy": test_accuracy_values,
    }


def plot_loss_curves(results: dict[str, list[float] | str]) -> None:
    """
    Plots the training and test loss and accuracy curves from the results of a train() function.

    Args:
        results (dict[str, list[float] | str]): The dictionary returned by the train() function containing loss values.
    """
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    train_accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title(f"Loss Curves for {results['model_name']}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, test_accuracy, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves for {results['model_name']}")
    plt.legend()
    plt.grid()


class TinyVGG(nn.Module):
    """A TinyVGG model for image classification.
    Will be used to train on the custom dataset created in this section."""

    def __init__(self, in_features: int, hidden_units: int, out_features: int) -> None:
        super().__init__()
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Calculate in_features dynamically via a dummy forward pass ---
        # Best practice: instead of hardcoding the flattened size (which breaks
        # whenever you change kernel_size, padding, or input resolution),
        # run a zero tensor through the conv stacks to let PyTorch compute the
        # output shape for us automatically.
        # torch.no_grad(): skip gradient tracking — this is just a shape probe.
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_features, 64, 64
            )  # [batch=1, C, H, W] — matches Resize(64,64) in transform
            dummy = self.conv_stack_2(self.conv_stack_1(dummy))
            # dummy shape after both conv stacks: [1, hidden_units, H_out, W_out]
            linear_in_features = dummy.flatten(start_dim=1).shape[1]
            # flatten(start_dim=1): collapse all dims except batch → [1, hidden_units*H_out*W_out]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=linear_in_features,  # auto-computed above
                out_features=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_stack_2(self.conv_stack_1(x)))
