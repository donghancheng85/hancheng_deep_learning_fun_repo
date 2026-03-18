import torch
from torch import nn
from typing import Callable


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device,
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

            # Accmulate the loss and accruacy values per batch
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
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    accruacy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device,
):
    """
    Performs one epoch of training on the given data_loader.

    Iterates over all batches, computes forward pass, loss, and accuracy,
    then backpropagates and updates model parameters.

    Args:
        model (nn.Module): The PyTorch model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training batches of (X, y).
        loss_fn (nn.Module): Loss function to compute per-batch training loss.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        accruacy_fn (Callable[[torch.Tensor, torch.Tensor], float]): Function that takes
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
        train_loss += loss_train_batch  # accumulate the training loss so we can calculate the average loss of the batches
        train_accuracy += accruacy_fn(y_true=y, y_pred=y_logits_train.argmax(dim=1))

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


def test_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    device: torch.device,
):
    """
    Performs one epoch of evaluation on the given data_loader.

    Runs the model in inference mode over all test batches, accumulating
    loss and accuracy, then prints the averaged results.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader providing test batches of (X, y).
        loss_fn (nn.Module): Loss function to compute per-batch test loss.
        accruacy_fn (Callable[[torch.Tensor, torch.Tensor], float]): Function that takes
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
            test_loss += loss_test_batch

            # 3. calculate the accuracy
            test_accuracy += accuracy_fn(
                y_true=y_test, y_pred=y_logits_test.argmax(dim=1)
            )

        # Calculate the test loss average per batch
        test_loss /= len(data_loader)

        # Calculate the test accuracy aversge per batch
        test_accuracy /= len(data_loader)

        print(f"\nTest loss: {test_loss:.5f}, Test accuracy: {test_accuracy:.2f}%\n")
