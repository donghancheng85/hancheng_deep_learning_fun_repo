import torch
from torch import nn
from typing import Callable, Tuple
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        train_loss += loss_train_batch.item()  # accumulate the training loss so we can calculate the average loss of the batches
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
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler to update the learning rate during training.
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
        print(f"Epoch {epoch + 1}/{epochs}")
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

        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

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


def train_for_summarywriter(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    accuracy_fn: Callable[[torch.Tensor, torch.Tensor], float],
    writer: SummaryWriter | None = None,
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
        scheduler (torch.optim.lr_scheduler._LRScheduler | None): Learning rate scheduler to update the learning rate during training.
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
        print(f"Epoch {epoch + 1}/{epochs}")
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

        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Append metrics for this epoch
        train_loss_values.append(train_loss)
        train_accuracy_values.append(train_accuracy)
        test_loss_values.append(test_loss)
        test_accuracy_values.append(test_accuracy)

        # Add experiment tracking with SummaryWriter
        if writer is not None:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                },
                global_step=epoch,
            )

    # Add model graph once after training (graph doesn't change between epochs)
    if writer is not None:
        # nn.MultiheadAttention has internal Python control flow that makes
        # torch.jit.trace produce structurally different graphs on the two
        # sanity-check passes — this is unfixable via strict=False.
        # Wrap in try/except so scalar metrics are always saved even if the
        # graph tab cannot be populated.
        model.eval()
        try:
            writer.add_graph(
                model=model,
                input_to_model=torch.randn(32, 3, 224, 224).to(device),
                use_strict_trace=False,
            )
        except Exception:
            print("[TensorBoard] Skipping add_graph — not compatible")
        writer.close()

    return {
        "model_name": model.__class__.__name__,
        "train_loss": train_loss_values,
        "train_accuracy": train_accuracy_values,
        "test_loss": test_loss_values,
        "test_accuracy": test_accuracy_values,
    }
