from collections.abc import Callable
import matplotlib.pyplot as plt
from pathlib import Path
from torch import Tensor
import torch
from torch.nn import Module, CrossEntropyLoss, MSELoss
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
        Saves a plot with two subplots: one for losses and one for accuracies over epochs.

        Args:
            path (Path): The path to save the plot to.
        """
        epoch_numbers = list(range(self.epoch_index))

        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.unicode_minus": False,
            "text.latex.preamble": r"\usepackage{lmodern}"
        })
       
        ax_loss.plot(epoch_numbers, self.training_losses, label="Training Loss", color="C0")
        ax_loss.plot(epoch_numbers, self.validation_losses, label="Validation Loss", color="C1")
        ax_loss.set_ylabel(r"$\mathrm{Loss}$")
        ax_loss.legend()
        ax_loss.grid(True)

        ax_acc.plot(epoch_numbers, self.training_accuracies, label="Training Accuracy", color="C0")
        ax_acc.plot(epoch_numbers, self.validation_accuracies, label="Validation Accuracy", color="C1")
        ax_acc.set_xlabel(r"$\mathrm{Epoch}$")
        ax_acc.set_ylabel(r"$\mathrm{Accuracy}$")
        ax_acc.legend()
        ax_acc.grid(True)

        fig.suptitle("Training and Validation Metrics", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

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
    if optimizer is not None:
        model.train()
    else: 
        model.eval()

    epoch_loss_train = 0.0
    for batch in dataloader:
        images, labels = batch

        if isinstance(loss, CrossEntropyLoss):  # Convert one-hot to class indices
            labels = labels.argmax(dim=1).long()
            labels_predicted = model(images)
            # Forward pass
            loss_batch = loss(labels_predicted, labels)
        elif isinstance(loss, MSELoss):
            labels = labels.float()
            labels_predicted = model(images)
            # Forward pass
            loss_batch = loss(labels_predicted, labels)
        else:
            # This is the case of informed regression
            target_classification = labels[:, :3].argmax(dim=1).long()
            target_regression = labels[:, 3:].float()

            labels_predicted_regression, labels_predicted_classification = model(images)

            # Forward pass
            loss_batch = loss(labels_predicted_classification, labels_predicted_regression, target_classification, target_regression)


        # Backward pass and optimization if an optimizer is provided
        if optimizer is not None:
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        # Accumulate loss
        epoch_loss_train += loss_batch.item()

    return epoch_loss_train / len(dataloader)

def compute_mse(network, dataloader):
    network.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = network(inputs)
            y_true.append(targets)
            y_pred.append(outputs)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    mse = torch.nn.functional.mse_loss(y_pred, y_true).item()
    return mse


def compute_accuracy(model: Module, dataloader: DataLoader, task_type: str) -> float:
    """
    Computes the accuracy (or other appropriate metric) of the model on the given data.

    Args:
        model (Module): The model to evaluate.
        dataloader (DataLoader): The dataloader providing the data.
        task_type (str): The task type (one of 'classification_binary', 'classification_multiclass', 'regression', 'informed_regression').

    Returns:
        float: The accuracy (or appropriate metric) of the model on the data.
    """
    prediction_count = 0
    correct_prediction_count = 0
    regression_loss = 0.0  # For tracking regression loss if needed
    
    for batch in dataloader:
        images, labels = batch

        # Get predictions from the model
        labels_predicted = model(images)

        if task_type in ["classification_binary", "classification_multiclass"]:
            # For classification tasks, take the class with the highest probability
            labels_predicted = labels_predicted.argmax(dim=1)
            
            # Convert one-hot labels to class indices if necessary
            if labels.ndimension() > 1:
                labels = labels.argmax(dim=1)
            
            correct_prediction_count += (labels_predicted == labels).sum().item()
            prediction_count += len(images) 

        elif task_type == "regression":
            # For regression tasks, no argmax, instead use Mean Squared Error or another regression metric
            regression_loss += torch.mean((labels_predicted - labels) ** 2).item()
            prediction_count += len(images)
        
        elif task_type == "informed_regression":
            # For informed regression, you get both regression and classification outputs
            output_regression, logits_classification = labels_predicted
            
            # For classification part (assuming softmax or similar output)
            labels_classification = labels[:, :3]  # Assuming the first 3 columns are for classification
            
            # Convert one-hot labels to class indices if necessary
            if labels_classification.ndimension() > 1:
                labels_classification = labels_classification.argmax(dim=1)

            labels_classification_pred = logits_classification.argmax(dim=1)

            correct_prediction_count += (labels_classification_pred == labels_classification).sum().item()

            # For regression part, calculate the regression loss
            labels_regression = labels[:, 3:]  # Assuming remaining part of labels is for regression
            regression_loss += torch.mean((output_regression - labels_regression) ** 2).item()

            prediction_count += len(images)

    # For regression, return the average loss
    if task_type in ["regression", "informed_regression"]:
        return regression_loss / prediction_count
    
    # For classification, return the accuracy
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
        TrainingSummary: An object containing the training summary, including loss and accuracy or mse.
    """
    summary = TrainingSummary(printing_interval_epochs=1)

    task_type = getattr(network, "task_type")

    for _ in range(epoch_count):
        
        # Training phase
        network.train()
        epoch_loss_training = compute_average_epoch_loss(
            network, training_dataloader, loss, optimizer
        )

        # Validation phase
        network.eval()
        epoch_loss_validation = compute_average_epoch_loss(
            network, validation_dataloader, loss
        )

        # Metrics depending on the task
        if getattr(network, "task_type", None) == "regression":
            epoch_metric_training = compute_mse(network, training_dataloader)
            epoch_metric_validation = compute_mse(network, validation_dataloader)
        else:
            epoch_metric_training = compute_accuracy(network, training_dataloader, task_type)
            epoch_metric_validation = compute_accuracy(network, validation_dataloader, task_type)

        # Save the results
        summary.append_epoch_summary(
            epoch_loss_training,
            epoch_metric_training,
            epoch_loss_validation,
            epoch_metric_validation,
        )

    return summary
