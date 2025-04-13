from galaxy_classification.networks.cnn import GalaxyCNNMLP, GalaxyClassificationCNNConfig

NetworkConfig = GalaxyClassificationCNNConfig

# TO-DO: BUILD LOSS FUNCTION

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
            assert False
        