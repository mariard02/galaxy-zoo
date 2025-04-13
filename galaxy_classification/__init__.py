from collections.abc import Callable
import matplotlib.pyplot as plt
from pathlib import Path
from torch import Tensor
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import colorful as cf

# Set the color scheme for the console output
cf.use_style('monokai')

def print_epoch_info(epoch_index, training_loss, training_accuracy, validation_loss, validation_accuracy, printing_interval_epochs):
    """
    Prints the summary of training and validation results for each epoch, 
    including the loss and accuracy, in a nicely formatted output with a 
    box around the epoch number.

    Args:
        epoch_index (int): The current epoch number.
        training_loss (float): The loss on the training data.
        training_accuracy (float): The accuracy on the training data.
        validation_loss (float): The loss on the validation data.
        validation_accuracy (float): The accuracy on the validation data.
        printing_interval_epochs (int): The interval at which to print the epoch information.
    """
    if (epoch_index + 1) % printing_interval_epochs == 0:
        # Create a formatted string with a box around the epoch
        epoch_str = f"{cf.purple}{cf.bold}Epoch {epoch_index + 1}{cf.reset}"

        # Create strings for training and validation metrics without changing background colors
        training_loss_str = f"Training Loss: {training_loss:.2e}"
        training_accuracy_str = f"Training Accuracy: {training_accuracy * 100.0:.2f}%"
        validation_loss_str = f"Validation Loss: {validation_loss:.2e}"
        validation_accuracy_str = f"Validation Accuracy: {validation_accuracy * 100.0:.2f}%"

        # Print with a box around the epoch
        print(f"{cf.bold}{cf.purple}{'=' * (len(epoch_str) + 4)}{cf.reset}")
        print(f"{cf.bold}{cf.purple}          {epoch_str}   {cf.reset}")
        print(f"{cf.bold}{cf.purple}{'=' * (len(epoch_str) + 4)}{cf.reset}")

        # Print the rest of the metrics
        print(f"{training_loss_str}")
        print(f"{training_accuracy_str}")
        print(f"{validation_loss_str}")
        print(f"{validation_accuracy_str}")
        print("\n")

class TrainingSummary:
    """
    A class to store and summarize the results of the training and validation
    processes during multiple epochs. It also provides functionality for
    plotting the results.

    Attributes:
        printing_interval_epochs (int): The interval at which to print epoch information.
        epoch_index (int): The current epoch number.
        training_losses (list): A list of training losses recorded per epoch.
        training_accuracies (list): A list of training accuracies recorded per epoch.
        validation_losses (list): A list of validation losses recorded per epoch.
        validation_accuracies (list): A list of validation accuracies recorded per epoch.
    """
    def __init__(self, printing_interval_epochs: int):
        """
        Initializes the TrainingSummary with empty lists and sets the printing interval and epoch index.

        Args:
            printing_interval_epochs (int): The interval at which to print epoch information.
        """
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.printing_interval_epochs = printing_interval_epochs
        self.epoch_index = 0

    def append_epoch_summary(
        self,
        training_loss: float,
        training_accuracy: float,
        validation_loss: float,
        validation_accuracy: float,
    ):
        """
        Appends the summary of a single epoch, including training and validation metrics.

        Args:
            training_loss (float): The loss on the training data.
            training_accuracy (float): The accuracy on the training data.
            validation_loss (float): The loss on the validation data.
            validation_accuracy (float): The accuracy on the validation data.
        """
        print_epoch_info(epoch_index=self.epoch_index, training_loss=training_loss, training_accuracy=training_accuracy, validation_loss=validation_loss, validation_accuracy=validation_accuracy, printing_interval_epochs=self.printing_interval_epochs)

        # Append metrics for this epoch
        self.training_losses.append(training_loss)
        self.training_accuracies.append(training_accuracy)
        self.validation_losses.append(validation_loss)
        self.validation_accuracies.append(validation_accuracy)
        self.epoch_index += 1

    def save_plot(self, path: Path):
        """
        Saves a plot of the training and validation losses and accuracies over epochs.

        Args:
            path (Path): The path to save the plot to.
        """
        # Create a figure and axis for loss plot
        figure, axes_loss = plt.subplots()

        epoch_numbers = list(range(self.epoch_index))
        axes_loss.plot(
            epoch_numbers, self.training_losses, label="training loss", color="C0"
        )
        axes_loss.plot(
            epoch_numbers, self.validation_losses, label="validation loss", color="C1"
        )
        axes_loss.set_xlabel("epoch")
        axes_loss.set_ylabel("loss")
        axes_loss.legend()

        # Create a second y-axis for accuracy plot
        axes_accuracy = axes_loss.twinx()
        axes_accuracy.plot(
            epoch_numbers,
            self.training_accuracies,
            label="training accuracy",
            color="C0",
            linestyle="dashed",
        )
        axes_accuracy.plot(
            epoch_numbers,
            self.validation_accuracies,
            label="validation accuracy",
            color="C1",
            linestyle="dashed",
        )
        axes_accuracy.set_ylabel("accuracy")
        axes_accuracy.legend()

        # Save the figure to the specified path
        figure.savefig(path, bbox_inches="tight")


def compute_average_epoch_loss(
    model: Module,
    dataloader: DataLoader,
    loss: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer | None = None,
):
    """
    Computes the average loss for a single epoch.

    Args:
        model (Module): The model to evaluate.
        dataloader (DataLoader): The dataloader providing the data.
        loss (Callable[[Tensor, Tensor], Tensor]): The loss function to use.
        optimizer (Optimizer | None): The optimizer used to update model parameters (if provided).

    Returns:
        float: The average loss for the epoch.
    """
    epoch_loss_train = 0.0
    for batch in dataloader:
        images, labels = batch["images"], batch["labels"]

        # Zero the gradients if an optimizer is provided
        if optimizer is not None:
            optimizer.zero_grad()

        # Forward pass
        labels_predicted = model(images)
        loss_batch = loss(labels_predicted, labels)

        # Backward pass and optimization if an optimizer is provided
        if optimizer is not None:
            loss_batch.backward()
            optimizer.step()

        # Accumulate loss
        epoch_loss_train += loss_batch.item()

    return epoch_loss_train / len(dataloader)


def compute_accuracy(model: Module, dataloader: DataLoader) -> float:
    """
    Computes the accuracy of the model on the given data.

    Args:
        model (Module): The model to evaluate.
        dataloader (DataLoader): The dataloader providing the data.

    Returns:
        float: The accuracy of the model on the data.
    """
    prediction_count = 0
    correct_prediction_count = 0
    for batch in dataloader:
        images, labels = batch["images"], batch["labels"]
        labels_predicted = model(images)

        # Get predicted labels by taking the class with the highest probability
        labels_predicted = labels_predicted.argmax(dim=1)
        correct_prediction_count += torch.sum(
            labels_predicted == labels
        ).item()
        prediction_count += len(batch["images"])

    return correct_prediction_count / prediction_count


def fit(
    network: Module,
    optimizer: Optimizer,
    loss: Callable[[Tensor, Tensor], Tensor],
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epoch_count: int,
) -> TrainingSummary:
    """
    Trains the model over multiple epochs and evaluates on both training and validation data.

    Args:
        network (Module): The neural network to train.
        optimizer (Optimizer): The optimizer used for training.
        loss (Callable[[Tensor, Tensor], Tensor]): The loss function.
        training_dataloader (DataLoader): The dataloader for the training data.
        validation_dataloader (DataLoader): The dataloader for the validation data.
        epoch_count (int): The number of epochs to train the model.

    Returns:
        TrainingSummary: An object containing the training summary, including loss and accuracy.
    """
    summary = TrainingSummary(printing_interval_epochs=1)
    for _ in range(epoch_count):
        # Training phase
        network.train()
        epoch_loss_training = compute_average_epoch_loss(
            network, training_dataloader, loss, optimizer
        )
        epoch_accuracy_training = compute_accuracy(network, training_dataloader)

        # Validation phase
        network.eval()
        epoch_loss_validation = compute_average_epoch_loss(
            network, validation_dataloader, loss
        )
        epoch_accuracy_validation = compute_accuracy(network, validation_dataloader)

        # Append the summary for the epoch
        summary.append_epoch_summary(
            epoch_loss_training,
            epoch_accuracy_training,
            epoch_loss_validation,
            epoch_accuracy_validation,
        )

    return summary
