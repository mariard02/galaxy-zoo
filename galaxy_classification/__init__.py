from collections.abc import Callable
import matplotlib.pyplot as plt
from pathlib import Path
from torch import Tensor
import torch
from torch.nn import Module, CrossEntropyLoss, MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import colorful as cf
import numpy as np
from galaxy_classification.networks import HierarchicalFocalLoss
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

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
        if training_accuracy is not None:
            training_accuracy_str = f"Training Accuracy: {training_accuracy * 100.0:.2f}%"
        validation_loss_str = f"Validation Loss: {validation_loss:.2e}"
        if validation_accuracy is not None:
            validation_accuracy_str = f"Validation Accuracy: {validation_accuracy * 100.0:.2f}%"

        # Print with a box around the epoch
        print(f"{cf.bold}{cf.purple}{'=' * (len(epoch_str) + 4)}{cf.reset}")
        print(f"{cf.bold}{cf.purple}          {epoch_str}   {cf.reset}")
        print(f"{cf.bold}{cf.purple}{'=' * (len(epoch_str) + 4)}{cf.reset}")

        # Print the rest of the metrics
        print(f"{training_loss_str}")
        if training_accuracy is not None:
            print(f"{training_accuracy_str}")
        print(f"{validation_loss_str}")
        if validation_accuracy is not None:
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
        If accuracies are None, only the losses plot is shown.

        Args:
            path (Path): The path to save the plot to.
        """
        epoch_numbers = list(range(self.epoch_index))

        # Check if accuracies are all None
        accuracies_none = all(acc is None for acc in self.training_accuracies + self.validation_accuracies)

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.unicode_minus": False,
            "text.latex.preamble": r"\usepackage{lmodern}"
        })

        if accuracies_none:
            # Only plot losses
            fig, ax_loss = plt.subplots(figsize=(8, 4))
            
            ax_loss.plot(epoch_numbers, self.training_losses, label="Training Loss", color="C0")
            ax_loss.plot(epoch_numbers, self.validation_losses, label="Validation Loss", color="C1")
            ax_loss.set_xlabel(r"$\mathrm{Epoch}$")
            ax_loss.set_ylabel(r"$\mathrm{Loss}$")
            ax_loss.legend()
            ax_loss.grid(True)

            fig.suptitle("Training and Validation Loss", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            # Plot both losses and accuracies
            fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            
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
        elif isinstance(loss, HierarchicalFocalLoss):
            labels = labels.float()
            labels_predicted = model(images)
            # Forward pass
            loss_batch = loss(labels_predicted, labels)
        else: 
            raise TypeError("Loss funtion not supported")

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

    # For regression, return the average loss
    if task_type == "informed_regression":
        return regression_loss / prediction_count
    
    if task_type == "regression":
        return None
    
    # For classification, return the accuracy
    return correct_prediction_count / prediction_count

def fit(
    network: Module,
    optimizer: Optimizer,
    loss: Callable[[Tensor, Tensor], Tensor],
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epoch_count: int,
    patience: int | None = None,   
    delta: float = 0.0,             
) -> TrainingSummary:
    """
    Trains the model over multiple epochs and evaluates on both training and validation data.
    Implements early stopping if patience is specified.

    Args:
        network (Module): The neural network to train.
        optimizer (Optimizer): The optimizer used for training.
        loss (Callable[[Tensor, Tensor], Tensor]): The loss function.
        training_dataloader (DataLoader): The dataloader for the training data.
        validation_dataloader (DataLoader): The dataloader for the validation data.
        epoch_count (int): The maximum number of epochs to train the model.
        patience (int | None): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in validation metric to qualify as an improvement.

    Returns:
        TrainingSummary: An object containing the training summary, including loss and accuracy or mse.
    """
    summary = TrainingSummary(printing_interval_epochs=1)

    task_type = getattr(network, "task_type")

    best_validation_loss = None
    epochs_without_improvement = 0

    for epoch in range(epoch_count):
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
        if task_type == "regression":
            epoch_metric_training = None
            epoch_metric_validation = None
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

        if best_validation_loss is None:
            best_validation_loss = epoch_loss_validation
            epochs_without_improvement = 0
        else:
            if epoch_loss_validation < best_validation_loss - delta:
                # Improvement: update best loss and reset counter
                best_validation_loss = epoch_loss_validation
                epochs_without_improvement = 0
            else:
                # No improvement: increment counter
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                    break

    return summary


# Function to plot ROC curves for model evaluation.
def plot_roc_curves(all_preds, all_labels, config, path: Path, class_names: list[str]):    
    """
    Generate and save ROC curves based on the model predictions and ground truth labels.
    
    Args:
        all_preds (Tensor): Model's predicted probabilities for each class (shape: [batch_size, num_classes]).
        all_labels (Tensor): True labels, either in one-hot encoding or class indices (shape: [batch_size]).
        config (EvaluationConfig): Configuration object that holds task type (binary or multiclass).
        path (Path): Path to save the plot.
        class_names (list[str]): List of class names for labeling the axes.
    """
    # Handle multiclass classification
    if config.task_type == "classification_multiclass":
        if all_labels.ndim > 1:
            all_labels = torch.argmax(all_labels, dim=1)  # Convert one-hot to class indices

        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure with a single axis

        num_classes = all_preds.shape[1]
        for i in range(num_classes):
            # Generate true binary labels for class i (1 if class i, else 0)
            y_true = (all_labels == i).numpy().astype(int)
            # Extract predicted probabilities for class i
            y_score = all_preds[:, i].numpy()

            # Plot ROC curve for class i
            RocCurveDisplay.from_predictions(
                y_true,
                y_score,
                name=class_names[i],
                ax=ax,  # Draw on the same axis
            )

        # Adjust plot settings
        ax.plot([0, 1], [0, 1], "k--")  # Random guess line
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.unicode_minus": False,
            "text.latex.preamble": r"\usepackage{lmodern}"
        })
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        fig.savefig(path, bbox_inches="tight")  # Save the plot to the specified path
        plt.close(fig)  # Close the plot to free memory

    # Handle binary classification
    elif config.task_type == "classification_binary":
        fig, ax = plt.subplots(figsize=(8, 6))

        RocCurveDisplay.from_predictions(
            all_labels.numpy(),
            all_preds.squeeze().numpy(),
            name="Binary Classification",
            ax=ax,
        )

        ax.plot([0, 1], [0, 1], "k--")  # Random guess line
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.grid()
        ax.legend()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

def plot_confusion_matrix(all_preds, all_labels, config, path: Path, class_names: list[str]):
    """
    Generate and save a confusion matrix for multiclass classification.

    Args:
        all_preds (Tensor): Model's predicted probabilities or logits (shape: [batch_size, num_classes]).
        all_labels (Tensor): True labels, either in one-hot encoding or class indices (shape: [batch_size]).
        config (EvaluationConfig): Configuration object that holds task type.
        path (Path): Path to save the plot.
        class_names (list[str]): List of class names for labeling the axes.
    """
    if config.task_type == "classification_multiclass":
        if all_labels.ndim > 1:
            all_labels = torch.argmax(all_labels, dim=1)  

        preds = torch.argmax(all_preds, dim=1) 

    elif config.task_type == "classification_binary":
        if all_preds.ndim > 1 and all_preds.shape[1] == 1:
            all_preds = all_preds.squeeze(dim=1)
        preds = (all_preds >= 0.5).long()
    else:    
        raise ValueError(f"Unsupported task type: {config.task_type}")
    
    print("computing confusion matrix")
    cm = confusion_matrix(all_labels.numpy(), preds.numpy(), normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{lmodern}"
    })

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_feature_histograms(properties_predicted, properties_true, xrange: tuple[float, float], path: Path, bins: int = 50, title: str = '', last_n_features: int | None = None, **kwargs):
    # Convertir tensores a numpy arrays si es necesario
    if hasattr(properties_predicted, 'numpy'):
        properties_predicted = properties_predicted.detach().numpy()
    if hasattr(properties_true, 'numpy'):
        properties_true = properties_true.detach().cpu().numpy()

    if properties_predicted.ndim == 1:
        properties_predicted = properties_predicted.reshape(-1, 1)
    if properties_true.ndim == 1:
        properties_true = properties_true.reshape(-1, 1)

    # Seleccionar solo las últimas N features si se especifica
    if last_n_features is not None:
        properties_predicted = properties_predicted[:, -last_n_features:]
        properties_true = properties_true[:, -last_n_features:]

    n_features = properties_predicted.shape[1]
    n_cols = min(n_features, 3)
    n_rows = (n_features - 1) // 3 + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_features * 3 + 5, 3.5 * n_rows))

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    plt.suptitle("Feature value distribution" + (f" ({title})" if title else ""))
    plt.subplots_adjust(wspace=0.5)

    for i, axis in enumerate(axes):
        if i >= n_features:
            axis.axis('off')
            continue

        axis.hist(properties_predicted[:, i], histtype="step", bins=bins, range=xrange,
                  label='Predicted' if i == 0 else "", **kwargs)
        axis.hist(properties_true[:, i], histtype="step", bins=bins, range=xrange,
                  label='True' if i == 0 else "", linestyle='--', **kwargs)

        axis.set_xlabel(f'Feature {-last_n_features + i + 1 if last_n_features else i + 1}')
        axis.set_ylabel('Frequency')

        if i == 0:
            axis.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)