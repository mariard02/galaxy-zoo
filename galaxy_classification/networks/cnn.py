from dataclasses import dataclass
from typing import Literal, Dict, List, Tuple, Optional
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
    task_type: Literal["classification_binary", "classification_multiclass", "regression", "informed_regression"]


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
       - Regression
       - Informed regression (regression conditioned on classification)
       - Hierarchical outputs

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
        task_type: Literal["classification_binary", "classification_multiclass", "regression", "informed_regression"],
        hierarchy_config: Optional[Dict] = None
    ):
        super().__init__()
        self.task_type = task_type
        VALID_TASK_TYPES = {"classification_binary", "classification_multiclass", "regression", "informed_regression"}
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
            Linear(mlp_hidden_unit_count * 2, mlp_hidden_unit_count * 4),  # Expand further
            ReLU(),
            Linear(mlp_hidden_unit_count * 4, mlp_hidden_unit_count * 2),  # Contract
            ReLU(),
            Linear(mlp_hidden_unit_count * 2, output_units * 4),  # Prepare for output
            ReLU(),
            Linear(output_units * 4, output_units * 2),
            ReLU(),
            Linear(output_units * 2, output_units),
            ReLU()
        )

        # Task-specific output heads
        if self.task_type == "informed_regression":
            # Two-head architecture: classification + regression
            self.classification_head = Linear(output_units, 2)
            self.regression_head = ConstrainedOutputLayer(output_units + 2, hierarchy_config)
        elif self.task_type == "regression":
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

        if self.task_type == "informed_regression":
            # Classification branch
            logits_cls = self.classification_head(features)
            probs_cls = F.softmax(logits_cls, dim=1)
            
            # Regression branch conditioned on classification
            combined = torch.cat([features, probs_cls], dim=1)
            output_reg = self.regression_head(combined)
            
            return output_reg, logits_cls
        elif self.task_type == "regression":
            return self.output_head(features)
        else:
            return self.output_head(features)


class ConstrainedOutputLayer(Module):
    """
    Hierarchical output layer that enforces parent-child relationships between outputs.

    Key Features:
    - Maintains probabilistic consistency in hierarchical outputs
    - Child class probabilities are constrained by parent probabilities
    - Supports arbitrary tree-like hierarchies

    Args:
        input_features: Number of input features
        hierarchy_config: Dictionary defining class hierarchy structure.
                         Format: {
                             'class1': (None, 2),  # (parent, num_subclasses)
                             'class2': ('class1.1', 2),  # Child of first subclass of class1
                             'class7': ('class1.2', 3)   # Child of second subclass of class1
                         }
    """
    def __init__(self, input_features: int, hierarchy_config: Dict[str, Tuple[str, int]]):
        super().__init__()
        self.hierarchy = hierarchy_config
        self.parent_relationships = {}
        
        # Create linear layers for each class group
        self.layers = ModuleDict()
        for class_name, (parent, num_subclasses) in hierarchy_config.items():
            self.layers[class_name] = Linear(input_features, num_subclasses)
            
            # Store parent relationships for forward pass
            if parent is not None:
                self.parent_relationships[class_name] = parent

        print("Subclasses generated:")
        for class_name in self.layers:
            print(f"{class_name}: {self.layers[class_name]}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with hierarchical constraints.

        Args:
            x: Input features tensor

        Returns:
            Tensor of concatenated probabilities respecting hierarchy constraints
        """
        outputs = {}
        # First pass: compute all base outputs
        for class_name in self.hierarchy:
            logits = self.layers[class_name](x)
            
            if class_name not in self.parent_relationships:
                # Root class - use sigmoid
                outputs[class_name] = torch.sigmoid(logits)
            else:
                # Store logits for dependent processing
                outputs[class_name] = logits
        
        # Second pass: process dependent classes
        for class_name, parent in self.parent_relationships.items():
            parent_name, parent_subclass = parent.split('.')
            parent_subclass = int(parent_subclass) - 1  # Convert to 0-based index
            
            # Get relevant parent probability
            parent_prob = outputs[parent_name][:, parent_subclass].unsqueeze(1)
            
            # Apply softmax and scale by parent probability
            outputs[class_name] = F.softmax(outputs[class_name], dim=1) * parent_prob
        
        # Concatenate all outputs in original order
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
    - Regression (MSELoss or HierarchicalFocalLoss)
    - Informed regression (combined classification + regression loss)
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
            "informed_regression"
        ],
        weight: Tensor = None,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        hierarchy_config: Optional[Dict[str, Tuple[str, int]]] = None,
        hierarchical_loss_params: Optional[Dict] = None
    ):
        VALID_TASK_TYPES = {
            "classification_binary",
            "classification_multiclass",
            "regression",
            "informed_regression"
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
        elif self.task_type == "informed_regression":
            self.classification_weight = classification_weight
            self.regression_weight = regression_weight
            self.classification_loss_fn = CrossEntropyLoss(weight=weight)
            self.regression_loss_fn = MSELoss()

    def get(self) -> Module:
        """Returns the appropriate loss function for the configured task."""
        if self.task_type == "informed_regression":
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
        Combines classification and regression losses for informed regression.

        Args:
            output_classification: Model's classification output
            output_regression: Model's regression output
            target_classification: True classification labels
            target_regression: True regression values

        Returns:
            Weighted sum of classification and regression losses
        """
        classification_loss = self.classification_loss_fn(output_classification, target_classification)
        regression_loss = self.regression_loss_fn(output_regression, target_regression)
        return self.classification_weight * classification_loss + self.regression_weight * regression_loss
    
class WeightedMSELoss(Module):
    """
    Weighted Mean Squared Error loss.

    Features:
    - Applies per-output weights
    - Automatically pads weights if fewer than 14 outputs
    - Maintains mean reduction

    Args:
        weights: Tensor of weights for each output dimension
    """
    def __init__(self, weights: torch.Tensor):
        super(WeightedMSELoss, self).__init__()
        if weights.numel() < 14:
            # Pad weights to 14 dimensions if needed
            padded_weights = torch.ones(14)
            padded_weights[:weights.numel()] = weights
            self.weights = padded_weights
        else:
            self.weights = weights

    def forward(self, prediction, target):
        """Compute weighted MSE loss."""
        squared_diff = (prediction - target) ** 2
        weighted_loss = self.weights.to(prediction.device) * squared_diff
        return weighted_loss.mean()