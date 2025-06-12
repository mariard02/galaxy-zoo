from galaxy_classification.networks.cnn import GalaxyCNNMLP, GalaxyClassificationCNNConfig, LossFunction, HierarchicalFocalLoss
from typing import Dict, Optional

NetworkConfig = GalaxyClassificationCNNConfig

def build_network(
    input_image_shape: tuple[int, int, int],
    config: NetworkConfig,
    hierarchy_config: Optional[Dict] = None
):
    """
    Builds and returns a neural network instance based on the given configuration.

    This function is designed to construct different types of networks depending
    on the configuration provided. Currently, it supports Galaxy CNN+MLP architectures
    for classification or regression tasks.

    Args:
        input_image_shape (tuple[int, int, int]): Shape of the input images (C, H, W).
        config (NetworkConfig): The configuration object specifying architecture and hyperparameters.
        hierarchy_config (dict, optional): Optional configuration for hierarchical classification.

    Returns:
        torch.nn.Module: A network instance built according to the provided configuration.

    Raises:
        ValueError: If the configuration type is not supported.
    """
    network_config = config

    # Check if the config is for a Galaxy CNN+MLP architecture
    if isinstance(network_config, GalaxyClassificationCNNConfig):
        return GalaxyCNNMLP(
            input_image_shape=input_image_shape,
            channel_count_hidden=network_config.channel_count_hidden,
            convolution_kernel_size=network_config.convolution_kernel_size,
            mlp_hidden_unit_count=network_config.mlp_hidden_unit_count,
            output_units=network_config.output_units,
            task_type=network_config.task_type,
            hierarchy_config=hierarchy_config  # Added support for hierarchical tasks
        )
    
    # Raise error for unsupported configuration types
    raise ValueError(f"Unsupported network config: {type(network_config)}")

        
def get_loss(config: NetworkConfig, weight=None, hierarchy_config=None):
    """
    Returns the appropriate loss function based on the network configuration.

    This function is designed to select and return a loss function depending
    on the task type specified in the network configuration. It supports class
    weighting and hierarchical classification if needed.

    Args:
        config (NetworkConfig): The overall configuration object, which contains the network configuration.
        weight (optional): Tensor of class weights for imbalanced classification problems.
        hierarchy_config (optional): Configuration dictionary for hierarchical classification, if applicable.

    Returns:
        torch.nn.Module: A PyTorch loss function appropriate for the task.

    Raises:
        ValueError: If the network configuration type is not supported.
    """
    match config.network:  # Pattern matching for future-proofing against multiple network types
        case GalaxyClassificationCNNConfig(task_type=task_type):
            return LossFunction(
                task_type=task_type,
                weight=weight,
                hierarchy_config=hierarchy_config
            ).get()
        case _:
            raise ValueError(f"Unsupported network config: {type(config.network)}")

