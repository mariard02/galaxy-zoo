from dataclasses import dataclass
from typing import Literal
import torch
from torch import Tensor
from torch.nn import (
    Conv2d,
    Flatten,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    AvgPool2d,
    functional,
    BCEWithLogitsLoss, 
    CrossEntropyLoss, 
    MSELoss,
    Module,
    LeakyReLU,
    AdaptiveAvgPool2d,
    BatchNorm2d
)
import torch.nn.functional as F

@dataclass
class GalaxyClassificationCNNConfig:
    """
    Configuration dataclass for the Galaxy CNN+MLP model.

    Attributes:
        channel_count_hidden (int): Number of hidden channels in convolutional layers.
        convolution_kernel_size (int): Kernel size for convolution operations.
        mlp_hidden_unit_count (int): Number of units in the first MLP layer.
        output_units (int): Number of output units (e.g., classes).
        task_type (Literal): Type of machine learning task (binary, multiclass, or regression).
    """
    channel_count_hidden: int
    convolution_kernel_size: int
    mlp_hidden_unit_count: int
    output_units: int
    task_type: Literal["classification_binary", "classification_multiclass", "regression", "informed_regression"]


class DoubleConvolutionBlock(Module):
    """
    A convolutional block with two Conv2D layers and optional residual connection.

    Args:
        channel_count_in (int): Number of input channels.
        channel_count_out (int): Number of output channels.
        channel_count_hidden (int): Number of hidden channels between convolutions.
        kernel_size (int): Kernel size for the convolutions.
    """
    def __init__(
        self,
        channel_count_in: int,
        channel_count_out: int,
        channel_count_hidden: int,
        kernel_size: int,
    ):
        super().__init__()

        self.conv1 = Conv2d(
            channel_count_in,
            channel_count_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = Conv2d(
            channel_count_hidden,
            channel_count_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.add_residual = channel_count_in == channel_count_out

    def forward(self, image: Tensor) -> Tensor:
        """
        Forward pass of the DoubleConvolutionBlock.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            Tensor: Output tensor after convolution and optional residual addition.
        """
        image_convolved = functional.relu(self.conv1(image))
        image_convolved = functional.relu(self.conv2(image_convolved))

        if self.add_residual:
            return image_convolved + image
        return image_convolved


class GalaxyCNNMLP(Module):
    """
    CNN + MLP model for image classification or regression.
    Produces raw logits — the loss function should apply the appropriate activation.

    Args:
        input_image_shape (tuple[int, int, int]): Input shape as (channels, height, width).
        channel_count_hidden (int): Hidden channels for CNN blocks.
        convolution_kernel_size (int): Size of convolution kernels.
        mlp_hidden_unit_count (int): Number of hidden units in the first MLP layer.
        output_units (int): Number of output units.
        task_type (Literal): Task type to determine loss function and optional output activation.
    """
    def __init__(
        self,
        input_image_shape: tuple[int, int, int],
        channel_count_hidden: int,
        convolution_kernel_size: int,
        mlp_hidden_unit_count: int,
        output_units: int,
        task_type: Literal["classification_binary", "classification_multiclass", "regression", "informed_regression"],
    ):
        super().__init__()

        self.task_type = task_type
        VALID_TASK_TYPES = {"classification_binary", "classification_multiclass", "regression", "informed_regression"}
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {task_type}. Must be one of {VALID_TASK_TYPES}")

        channels, height, width = input_image_shape

        self.cnn = Sequential(
            DoubleConvolutionBlock(
                channels,
                channel_count_hidden,
                channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            ReLU(),
            AvgPool2d(kernel_size=2),
            DoubleConvolutionBlock(
                channel_count_hidden,
                channel_count_hidden,
                channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            ReLU(),
            AvgPool2d(kernel_size=2),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            flattened_size = self.cnn(dummy_input).view(1, -1).shape[1]

        self.mlp = Sequential(
            Flatten(),
            Linear(flattened_size, mlp_hidden_unit_count),
            ReLU(),
            Linear(mlp_hidden_unit_count, mlp_hidden_unit_count//2),
            ReLU(),
            Linear(mlp_hidden_unit_count//2, mlp_hidden_unit_count),
            ReLU(),
            Linear(mlp_hidden_unit_count, output_units, bias=True),
        )

        # Some variables for the case of informed regression
        if self.task_type == "informed_regression":
            self.classification_layer = Linear(output_units, 3) # Output of the classification layer: 3 possible classes
            self.regression_layer = Linear(3 + output_units, output_units) # Regression layer: stack classification and mlp

        # No activation function here; it's handled by the loss function
        self.output_units = output_units

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through CNN and MLP. Returns raw logits.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Logits or raw predictions.
        """
        x = self.cnn(x)
        x = self.mlp(x)

        if self.task_type == "informed_regression":
            logits_classification = self.classification_layer(x)
            output_classification = F.softmax(logits_classification, dim=1)  # Returns the most probability of each class

            # Concatenate the output from the classification with the MLP output
            combined_features = torch.cat((x, output_classification), dim=1)

            # Regression
            output_regression = self.regression_layer(combined_features)
            return output_regression, logits_classification # We need both the outputs for calculating the loss
        return x

class LossFunction:
    """
    Provides the appropriate loss function based on the task type.

    Args:
        task_type (Literal): The type of task. Must be one of:
            - "classification_binary"
            - "classification_multiclass"
            - "regression"
            - "informed_regression"
    """

    def __init__(self, task_type: Literal["classification_binary", "classification_multiclass", "regression", "informed_regression"], weight: Tensor, classification_weight: float = 1.0,
        regression_weight: float = 1.0):
        VALID_TASK_TYPES = {"classification_binary", "classification_multiclass", "regression", "informed_regression"}
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {task_type}. Must be one of {VALID_TASK_TYPES}")

        self.task_type = task_type

        if self.task_type == "classification_binary":
            # Assumes model output uses Sigmoid activation
            self.loss_fn = BCEWithLogitsLoss()
        elif self.task_type == "classification_multiclass":
            # Assumes model output uses raw logits (no Softmax)
            self.loss_fn = CrossEntropyLoss(weight = weight)
        elif self.task_type == "regression":
            self.loss_fn = MSELoss()
        elif self.task_type == "informed_regression":
            # For informed regression, we'll combine classification and regression losses
            self.classification_weight = classification_weight
            self.regression_weight = regression_weight

            # Loss functions for classification and regression
            self.classification_loss_fn = CrossEntropyLoss(weight=weight)
            self.regression_loss_fn = MSELoss()

    def get(self) -> Module:
        """
        Returns the appropriate loss function for the task.

        Returns:
            nn.Module: The loss function.
        """
        if self.task_type == "informed_regression":
            # Return a custom loss function that combines classification and regression losses
            return self.informed_regression_loss
        return self.loss_fn

    def informed_regression_loss(
        self, 
        output_classification: Tensor, 
        output_regression: Tensor, 
        target_classification: Tensor, 
        target_regression: Tensor
    ) -> Tensor:
        """
        Combines classification and regression losses into a single loss for informed regression.

        Args:
            output_classification (Tensor): The raw logits for classification.
            output_regression (Tensor): The predicted values for regression.
            target_classification (Tensor): The true labels for classification.
            target_regression (Tensor): The true values for regression.

        Returns:
            Tensor: The combined loss.
        """
        # Classification loss
        classification_loss = self.classification_loss_fn(output_classification, target_classification)

        # Regression loss
        regression_loss = self.regression_loss_fn(output_regression, target_regression)

        # Combined loss
        total_loss = self.classification_weight * classification_loss + self.regression_weight * regression_loss
        return total_loss