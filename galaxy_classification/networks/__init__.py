from galaxy_classification.networks.cnn import GalaxyCNNMLP, GalaxyClassificationCNNConfig, LossFunction, WeightedMSELoss, HierarchicalFocalLoss
from typing import Dict, Optional

NetworkConfig = GalaxyClassificationCNNConfig

def build_network(
    input_image_shape: tuple[int, int, int],
    config: NetworkConfig,
    hierarchy_config: Optional[Dict] = None
):
    network_config = config
    
    # Verificamos si es una configuración de CNN
    if isinstance(network_config, GalaxyClassificationCNNConfig):
        return GalaxyCNNMLP(
            input_image_shape=input_image_shape,
            channel_count_hidden=network_config.channel_count_hidden,
            convolution_kernel_size=network_config.convolution_kernel_size,
            mlp_hidden_unit_count=network_config.mlp_hidden_unit_count,
            output_units=network_config.output_units,
            task_type=network_config.task_type,
            hierarchy_config=hierarchy_config  # Añadimos esto sin romper lo existente
        )
    
    raise ValueError(f"Unsupported network config: {type(network_config)}")
        
def get_loss(config: NetworkConfig, weight = None, hierarchy_config = None):
    match config.network: # Protection in case in the future I add different networks.
        case GalaxyClassificationCNNConfig(task_type=task_type):
            return LossFunction(task_type=task_type, weight=weight, hierarchy_config=hierarchy_config).get()
        case _:
            raise ValueError(f"Unsupported network config: {type(config.network)}")
