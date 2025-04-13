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
)


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
    task_type: Literal["classification_binary", "classification_multiclass", "regression"]


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

    Args:
        input_image_shape (tuple[int, int, int]): Input shape as (channels, height, width).
        channel_count_hidden (int): Hidden channels for CNN blocks.
        convolution_kernel_size (int): Size of convolution kernels.
        mlp_hidden_unit_count (int): Number of hidden units in the first MLP layer.
        output_units (int): Number of output units.
        task_type (Literal): Task type to determine activation function.
    """
    def __init__(
        self,
        input_image_shape: tuple[int, int, int],
        channel_count_hidden: int,
        convolution_kernel_size: int,
        mlp_hidden_unit_count: int,
        output_units: int,
        task_type: Literal["classification_binary", "classification_multiclass", "regression"],
    ):
        super().__init__()

        self.task_type = task_type
        channels, height, width = input_image_shape

        self.cnn = Sequential(
            DoubleConvolutionBlock(
                channels,
                channel_count_hidden,
                channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            AvgPool2d(kernel_size=2),
            DoubleConvolutionBlock(
                channel_count_hidden,
                channel_count_hidden,
                channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            AvgPool2d(kernel_size=2),
            DoubleConvolutionBlock(
                channel_count_hidden,
                channel_count_hidden,
                channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            AvgPool2d(kernel_size=2),
        )

        # Automatically calculate output size after CNN block
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            flattened_size = self.cnn(dummy_input).view(1, -1).shape[1]

        self.mlp = Sequential(
            Flatten(),
            Linear(flattened_size, mlp_hidden_unit_count),
            ReLU(),
            Linear(mlp_hidden_unit_count, mlp_hidden_unit_count // 2),
            ReLU(),
            Linear(mlp_hidden_unit_count // 2, mlp_hidden_unit_count // 2),
            ReLU(),
            Linear(mlp_hidden_unit_count // 2, output_units),
        )

        # Define task-specific output activation
        if self.task_type == "classification_binary":
            self.activation = Sigmoid()
        elif self.task_type == "classification_multiclass":
            self.activation = Softmax(dim=1)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the CNN and MLP blocks.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Model output after passing through CNN, MLP, and activation function.
        """
        x = self.cnn(x)
        x = self.mlp(x)
        if self.activation:
            x = self.activation(x)
        return x
