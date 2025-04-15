from galaxy_classification.networks.cnn import GalaxyCNNMLP, GalaxyClassificationCNNConfig, LossFunction

NetworkConfig = GalaxyClassificationCNNConfig

def build_network(
    input_image_shape: tuple[int, int, int],
    config: NetworkConfig,
):
    match config: # Protection in case in the future I add different networks.
        case GalaxyClassificationCNNConfig(
            channel_count_hidden=channel_count_hidden,
            convolution_kernel_size=convolution_kernel_size,
            mlp_hidden_unit_count=mlp_hidden_unit_count,
            output_units=output_units,
            task_type=task_type,
        ):
            return GalaxyCNNMLP(
                input_image_shape=input_image_shape,
                channel_count_hidden=channel_count_hidden,
                convolution_kernel_size=convolution_kernel_size,
                mlp_hidden_unit_count=mlp_hidden_unit_count,
                output_units=output_units,
                task_type=task_type,
            )
        case _:
            raise ValueError(f"Unsupported network config: {type(config.network)}")
        
def get_loss(config: NetworkConfig):
    match config.network:
        case GalaxyClassificationCNNConfig(task_type=task_type):
            return LossFunction(task_type=task_type).get()
        case _:
            raise ValueError(f"Unsupported network config: {type(config.network)}")
