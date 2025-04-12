from dataclasses import dataclass
from typing import Literal
from torch.nn import Flatten, Linear, Module, ReLU, Sequential, Sigmoid, Softmax

class GalaxyMLP(Module):
    """
    Multi-Layer Perceptron (MLP) for galaxy classification or regression tasks.

    This class builds a neural network that can be used for different types of tasks:
    - Binary classification (output: 1 unit, Sigmoid activation)
    - Multi-class classification (output: 'output_units' units, Softmax activation)
    - Regression (output: 'output_units' units, no activation function)

    Parameters:
    - image_input_shape (tuple[int, int]): The shape of the input image (height, width).
    - mlp_hidden_unit_count (int): Number of units in the first hidden layer.
    - output_units (int): Number of output units corresponding to the task (1 for binary classification, 
      number of classes for multi-class classification, or any value depending on the regression problem).
    - task_type (Literal["classification_binary", "classification_multiclass", "regression"]): 
      Specifies the task type: 'classification_binary', 'classification_multiclass', or 'regression'.
    """
    
    def __init__(
        self,
        image_input_shape: tuple[int, int],
        mlp_hidden_unit_count: int,
        output_units: int,
        task_type: Literal["classification_binary", "classification_multiclass", "regression"]
    ):
        """
        Initialize the GalaxyMLP model.

        Args:
        - image_input_shape (tuple[int, int]): The input shape of the image.
        - mlp_hidden_unit_count (int): Number of units in the first hidden layer.
        - output_units (int): Number of units in the output layer.
        - task_type (str): Specifies the task type, which affects the activation function.
        """
        super().__init__()

        self.task_type = task_type

        # Build the MLP architecture
        self.network = Sequential(
            Flatten(),  # Flatten the image input into a vector
            Linear(image_input_shape[0] * image_input_shape[1], mlp_hidden_unit_count),  # First hidden layer
            ReLU(),  # ReLU activation
            Linear(mlp_hidden_unit_count, mlp_hidden_unit_count // 2),  # Second hidden layer
            ReLU(),  # ReLU activation
            Linear(mlp_hidden_unit_count // 2, mlp_hidden_unit_count // 2),  # Third hidden layer
            ReLU(),  # ReLU activation
            Linear(mlp_hidden_unit_count // 2, output_units),  # Output layer
        )

        # Choose activation based on the task type
        if self.task_type == "classification_binary":
            self.activation = Sigmoid()  # Use Sigmoid for binary classification
        elif self.task_type == "classification_multiclass":
            self.activation = Softmax(dim=1)  # Use Softmax for multiclass classification
        else:
            self.activation = None  # No activation for regression

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (Tensor): Input tensor (batch of images).

        Returns:
        - Tensor: Output tensor after passing through the network and activation function (if any).
        """
        # Pass the input through the MLP network layers
        x = self.network(x)
        
        # Apply activation function if specified (Sigmoid, Softmax, or None)
        if self.activation:
            x = self.activation(x)
        
        return x
