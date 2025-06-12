from dataclasses import dataclass
from typing import Literal, Dict, List, Tuple, Optional, Union
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
    BatchNorm2d, 
    ModuleDict
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
    task_type: Literal["classification_binary", "classification_multiclass", "regression"]

class DoubleConvolutionBlock(Module):
    """
    A convolutional block with two Conv2D layers and optional residual connection.

    Features:
    - Two convolutional layers with batch normalization
    - ReLU activation functions
    - Optional residual connection when input/output channels match
    - Kernel size padding maintains spatial dimensions

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
            padding=kernel_size // 2,  # Maintains spatial dimensions
        )
        self.conv2 = Conv2d(
            channel_count_hidden,
            channel_count_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.bn1 = BatchNorm2d(channel_count_hidden)
        self.bn2 = BatchNorm2d(channel_count_out)

        # Enable residual connection if input/output channels match
        self.add_residual = channel_count_in == channel_count_out

    def forward(self, image: Tensor) -> Tensor:
        """
        Forward pass of the DoubleConvolutionBlock.

        Args:
            image (Tensor): Input image tensor of shape (batch, channels, height, width)

        Returns:
            Tensor: Output tensor after convolution and optional residual addition.
        """
        image_convolved = F.relu(self.bn1(self.conv1(image)))
        image_convolved = F.relu(self.bn2(self.conv2(image_convolved)))

        if self.add_residual:
            return image_convolved + image
        return image_convolved


class GalaxyCNNMLP(Module):
    """
    Main CNN+MLP model for galaxy image analysis with flexible output heads.

    Architecture:
    1. CNN feature extractor (DoubleConvolutionBlocks with pooling)
    2. MLP feature processor
    3. Task-specific output heads:
       - Classification (binary/multiclass)
       - Hierarchical outputs for regression

    Args:
        input_image_shape: Tuple of (channels, height, width)
        channel_count_hidden: Number of channels in hidden layers
        convolution_kernel_size: Size of convolutional kernels
        mlp_hidden_unit_count: Base size of MLP hidden layers
        output_units: Number of output units
        task_type: Type of prediction task
        hierarchy_config: Configuration for hierarchical outputs (if applicable)
    """
    def __init__(
        self,
        input_image_shape: tuple[int, int, int],
        channel_count_hidden: int,
        convolution_kernel_size: int,
        mlp_hidden_unit_count: int,
        output_units: int,
        task_type: Literal["classification_binary", "classification_multiclass", "regression"],
        hierarchy_config: Optional[Dict] = None
    ):
        super().__init__()
        self.task_type = task_type
        VALID_TASK_TYPES = {"classification_binary", "classification_multiclass", "regression"}
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {task_type}")

        channels, height, width = input_image_shape

        # CNN feature extractor
        self.cnn = Sequential(
            DoubleConvolutionBlock(
                channels, channel_count_hidden, channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            ReLU(),
            AvgPool2d(kernel_size=2),  # Downsample by factor of 2
            DoubleConvolutionBlock(
                channel_count_hidden, channel_count_hidden, channel_count_hidden,
                kernel_size=convolution_kernel_size,
            ),
            ReLU(),
            AvgPool2d(kernel_size=2),  # Downsample by factor of 2
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            flattened_size = self.cnn(dummy_input).view(1, -1).shape[1]

        # MLP feature processor with expanding/contracting architecture
        self.mlp = Sequential(
            Flatten(),
            Linear(flattened_size, mlp_hidden_unit_count),
            ReLU(),
            Linear(mlp_hidden_unit_count, mlp_hidden_unit_count * 2),  # Expand
            ReLU(),
            Linear(mlp_hidden_unit_count * 2, output_units * 2),  # Prepare for output
            ReLU(),
            Linear(output_units * 2, output_units),
            ReLU()
        )

        # Task-specific output heads
        if self.task_type == "regression":
            self.output_head = ConstrainedOutputLayer(output_units, hierarchy_config)
        else:
            self.output_head = Linear(output_units, output_units)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with task-specific output processing.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            For informed_regression: tuple of (regression_output, classification_logits)
            For other tasks: single output tensor
        """
        x = self.cnn(x)
        features = self.mlp(x)

        return self.output_head(features)

class ConstrainedOutputLayer(Module):
    """
    Hierarchical output layer that enforces parent-child relationships between outputs.

    Key Features:
    - Maintains probabilistic consistency in hierarchical outputs
    - Child class probabilities are constrained by one or more parent probabilities
    - Supports arbitrary tree-like hierarchies with shared subtrees

    Args:
        input_features: Number of input features
        hierarchy_config: Dictionary defining class hierarchy structure.
                         Format: {
                             'class1': (None, 2),  # (parent(s), num_subclasses)
                             'class2': ('class1.1', 2),
                             'class7': ('class1.2', 3),
                             'class8': (['class1.1', 'class7.3'], 2)  # Multiple parents
                         }
    """
    def __init__(self, input_features: int, hierarchy_config: Dict[str, Tuple[Union[str, List[str], None], int]]):
        super().__init__()
        self.hierarchy = hierarchy_config
        self.parent_relationships = {}
        
        # Create linear layers for each class group
        self.layers = ModuleDict()
        for class_name, (parents, num_subclasses) in hierarchy_config.items():
            self.layers[class_name] = Linear(input_features, num_subclasses)
            
            # Store parent relationships for forward pass
            if parents is not None:
                self.parent_relationships[class_name] = parents

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hierarchical constraints.

        Args:
            x: Input features tensor

        Returns:
            Tensor of concatenated probabilities respecting hierarchy constraints
        """
        outputs = {}

        # First pass: compute base logits or probabilities
        for class_name in self.hierarchy:
            logits = self.layers[class_name](x)
            if class_name not in self.parent_relationships:
                # Root class - independent
                outputs[class_name] = torch.sigmoid(logits)
            else:
                # Save logits for now
                outputs[class_name] = logits

        # Second pass: apply constraints for children
        for class_name, parents in self.parent_relationships.items():
            # Ensure list format for consistency
            if isinstance(parents, str):
                parents = [parents]

            parent_probs = []
            for parent in parents:
                parent_name, parent_subclass = parent.split('.')
                parent_subclass = int(parent_subclass) - 1  # 1-based to 0-based
                parent_prob = outputs[parent_name][:, parent_subclass].unsqueeze(1)
                parent_probs.append(parent_prob)

            # Combine parent probabilities (sum)
            joint_parent_prob = torch.stack(parent_probs, dim=0).sum(dim=0)

            # Optional: clamp to [0, 1] to avoid exceeding 1
            joint_parent_prob = torch.clamp(joint_parent_prob, max=1.0)

            # Apply softmax to logits and scale by joint parent prob
            outputs[class_name] = F.softmax(outputs[class_name], dim=1) * joint_parent_prob

        # Concatenate all outputs in hierarchy order
        return torch.cat([outputs[class_name] for class_name in self.hierarchy], dim=1)

    
class HierarchicalFocalLoss(Module):
    """
    Custom loss function for hierarchical outputs combining:
    - Focal loss properties (focus on hard examples)
    - Hierarchical weighting (balance between class groups)
    - MSE base loss

    Args:
        hierarchy_config: Same as ConstrainedOutputLayer
        alpha: Balancing parameter for focal loss
        gamma: Focusing parameter for focal loss
        class_weights: Optional weights for each class group
    """
    def __init__(self, hierarchy_config: Dict[str, Tuple[str, int]], alpha=0.8, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = MSELoss(reduction='none')  # Keep per-element losses
        
        # Calculate weights if not provided
        if class_weights is None:
            total_classes = sum(num_subclasses for _, num_subclasses in hierarchy_config.values())
            class_weights = {
                class_name: num_subclasses/total_classes 
                for class_name, (_, num_subclasses) in hierarchy_config.items()
            }
        
        self.class_weights = class_weights
        self.class_ranges = self._calculate_class_ranges(hierarchy_config)

    
    def _calculate_class_ranges(self, hierarchy_config):
        """Calculates column ranges for each class group in concatenated output."""
        ranges = {}
        start = 0
        for class_name, (_, num_subclasses) in hierarchy_config.items():
            ranges[class_name] = (start, start + num_subclasses)
            start += num_subclasses
        return ranges
    
    def forward(self, preds, targets):
        """
        Compute hierarchical focal loss.

        Args:
            preds: Model predictions
            targets: Ground truth values

        Returns:
            Weighted loss value
        """
        losses = self.mse(preds, targets)
        
        # Focal weighting based on prediction confidence
        focal_weights = torch.abs(preds.detach() - targets).pow(self.gamma)
        losses = self.alpha * focal_weights * losses
        
        # Compute weighted losses per class group
        total_loss = 0.0
        for class_name, (start, end) in self.class_ranges.items():
            class_loss = losses[:, start:end].mean()
            total_loss += self.class_weights[class_name] * class_loss
        
        return total_loss

class LossFunction:
    """
    Factory class providing appropriate loss functions based on task type.

    Supports:
    - Binary classification (BCEWithLogitsLoss)
    - Multiclass classification (CrossEntropyLoss)
    - Hierarchical outputs (HierarchicalFocalLoss)

    Args:
        task_type: Type of prediction task
        weight: Class weights for classification tasks
        classification_weight: Weight for classification in informed regression
        regression_weight: Weight for regression in informed regression
        hierarchy_config: Configuration for hierarchical outputs
        hierarchical_loss_params: Parameters for hierarchical loss
    """

    def __init__(
        self,
        task_type: Literal[
            "classification_binary",
            "classification_multiclass",
            "regression",
        ],
        weight: Tensor = None,
        hierarchy_config: Optional[Dict[str, Tuple[str, int]]] = None,
        hierarchical_loss_params: Optional[Dict] = None
    ):
        VALID_TASK_TYPES = {
            "classification_binary",
            "classification_multiclass",
            "regression",
        }
        
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {task_type}. Must be one of {VALID_TASK_TYPES}")

        self.task_type = task_type
        self.hierarchy_config = hierarchy_config
        self.hierarchical_loss_params = hierarchical_loss_params or {}
        
        # Initialize appropriate loss functions
        if self.task_type == "classification_binary":
            self.loss_fn = BCEWithLogitsLoss(weight=weight)
        elif self.task_type == "classification_multiclass":
            self.loss_fn = CrossEntropyLoss(weight=weight)
        elif self.task_type == "regression":
            if not hierarchy_config:
                raise ValueError("hierarchy_config must be provided for hierarchical task type")
            self.loss_fn = HierarchicalFocalLoss(
                hierarchy_config=hierarchy_config,
                **self.hierarchical_loss_params
            )
        else:
            raise ValueError("Loss function not supported")


    def get(self) -> Module:
        """Returns the appropriate loss function for the configured task."""
        return self.loss_fn
    