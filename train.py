import dacite
import logging
import os
import shutil
import simple_parsing
import torch
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import colorful as cf
from galaxy_classification.data import *
from galaxy_classification.networks import *
import galaxy_classification
from asciiart import *


cf.use_style('monokai')

# Dataclass to load the data from the run
@dataclass
class TrainingCli:
    """
    Attributes:
    run_name (str): Unique name for the training run (used for logging, saving models, etc.).
    no_config_edit (bool): If True, disables manual editing of configuration after parsing.
    """
    run_name: str
    no_config_edit: bool = False

@dataclass
class TrainingConfig: 
    epoch_count: int
    batch_size: int
    learning_rate: float

    validation_fraction: float

    network: NetworkConfig

def load_config(path: Path) -> TrainingConfig:
    """
    Loads the configuration from a YAML file and converts it into a `TrainingConfig` object.

    This function reads the YAML file at the given path, extracts the "training" section,
    and maps the data to a `TrainingConfig` object using `dacite.from_dict`.

    Args:
        path (Path): The path to the YAML configuration file.

    Returns:
        TrainingConfig: A `TrainingConfig` object containing the parsed configuration data.
    """
    # Open the configuration file and load its contents
    with open(path) as config_file:
        # Load the "training" section from the YAML file and map it to a TrainingConfig object
        return dacite.from_dict(TrainingConfig, yaml.safe_load(config_file)["training"])

    
def prepare_config(
    output_path: Path, default_path: Path, run_name: str, allow_config_edit: bool
) -> TrainingConfig:
    """
    Prepares the configuration file for a new training run. This function copies the default
    configuration to the specified output path and optionally allows the user to edit the config
    before returning a `TrainingConfig` object.

    Args:
        output_path (Path): The path where the configuration file will be saved for this run.
        default_path (Path): The path to the default configuration file to be copied.
        run_name (str): The name of the current run, used to personalize paths and logs.
        allow_config_edit (bool): Whether the user is allowed to manually edit the configuration
                                   before starting the training.

    Returns:
        TrainingConfig: A `TrainingConfig` object populated with the values from the configuration file.
    """
    # Ensure the parent directory of the output path exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Copy the default configuration file to the output path
    print(f"Copying {default_path} to {output_path}")
    shutil.copy(default_path, output_path)

    # If the user is allowed to edit the config, prompt them to do so
    if allow_config_edit:
        _ = input(
            f"Please edit the config in outputs/{run_name}/config.yaml"
            " to set the parameters for this run.\n"
            "After, press enter to continue."
        )
    
    # Load and return the configuration from the output path
    return load_config(output_path)


def build_transform(image_dir: Path, label_path: Path) -> torch.nn.Module:
    """
    Builds the transformation pipeline for preprocessing the image data.
    This pipeline includes resizing the images, cropping them, converting them to tensors,
    and applying automatic normalization based on the provided image and label paths.

    Args:
        image_dir (Path): The directory where the images are stored.
        label_path (Path): The path to the labels file (CSV format or similar) corresponding to the images.

    Returns:
        torch.nn.Module: Transform pipeline, which includes:
            - Resizing the image to 128x128 pixels.
            - Cropping a 64x64 region from the center of the resized image.
            - Converting the image to a tensor, which also scales pixel values to the [0, 1] range.
            - Applying normalization using the `AutoNormalizeTransform`, which calculates and applies 
              mean and standard deviation based on the images in the given directories.
    """
    return transforms.Compose([
        # Resize the image to 128x128 pixels
        transforms.Resize((128, 128)),

        # Crop a 64x64 region from the center of the image
        transforms.Lambda(lambda x: transforms.functional.crop(x, 32, 32, 64, 64)),

        transforms.RandomHorizontalFlip(),
        
        transforms.RandomVerticalFlip(),

        # Convert the image to a tensor and scale pixel values to [0, 1]
        transforms.ToTensor(),

        # Apply automatic normalization based on image and label paths provided
        AutoNormalizeTransform(
            image_dir,  # Pass the image directory path
            label_path,  # Pass the label file path
        ),

    ])

def save_hyperparameters(path: Path, config: NetworkConfig):
    """
    Saves the network configuration (hyperparameters) to a YAML file.

    This function converts the `NetworkConfig` object into a dictionary using `asdict`,
    then writes it to a YAML file at the specified path. The file will contain the hyperparameters
    used to configure the neural network for the training run.

    Args:
        path (Path): The path where the hyperparameters YAML file will be saved.
        config (NetworkConfig): The `NetworkConfig` object containing the hyperparameters to be saved.

    Returns:
        None
    """
    # Open the specified path in write mode
    with open(path, "w") as hyperparameter_cache:
        # Convert the NetworkConfig object into a dictionary and write it as YAML
        yaml.dump(asdict(config), hyperparameter_cache)


def main():
    """
    Main entry point of the script.

    Parses command-line arguments into a TrainingCli object and performs any 
    setup or execution logic required to start the training run.
    """
    # Create a minimal parser without default help sections
    parser = simple_parsing.ArgumentParser(add_help=True, description="Train the galaxy classifier.")
    # Add the dataclass
    parser.add_arguments(TrainingCli, dest="cli")
    # Parse arguments
    args = parser.parse_args()
    # Unpack the dataclass
    cli: TrainingCli = args.cli

    image_dir = Path("data/images/images_training_rev1")
    label_path = Path("data/exercise_4/labels.csv")

    print("\n" + cf.purple(generate_title_string()) + "\n") 
    
    print_divider()
    print(f"Run name: {cf.purple(cli.run_name)}" + "\n")

    config = prepare_config(
        Path(f"outputs/{cli.run_name}/config.yaml"),
        Path("config_default.yaml"),
        cli.run_name,
        not cli.no_config_edit,
    )

    # Load a dataset with the data
    print("\nLoading the dataset. \n")
    transform = build_transform(image_dir=image_dir, label_path=label_path)
    galaxy_dataset = load_image_dataset(image_dir, label_path, transform=transform)   

    print("Preprocessing the data. \n")
    preprocessor = GalaxyPreprocessor()
    galaxy_preprocessed = preprocessor.apply_preprocessing(galaxy_dataset)

    split_dataloader = SplitGalaxyDataLoader(
        galaxy_preprocessed,
        config.validation_fraction,
        config.batch_size,
    )

    print("Building the CNN. \n")
    network = build_network(
        galaxy_preprocessed.image_shape(),
        config.network,
    )

    optimizer = AdamW(network.parameters(), lr = config.learning_rate, weight_decay=1.e-5) # The optimizer does not depend on the task
   
    weights_binary = GalaxyWeightsClassification(galaxy_dataset)
    weights = weights_binary.get_weights()

    loss = get_loss(config=config)

    print_divider()

    print("Training... \n")

    training_summary = galaxy_classification.fit(
        network,
        optimizer,
        loss,
        split_dataloader.training_dataloader,
        split_dataloader.validation_dataloader,
        config.epoch_count,
    )

    print(
        f"Saving training summary plots to outputs/{cli.run_name}/plots/training_summary.pdf"
    )

    os.makedirs(f"outputs/{cli.run_name}/plots", exist_ok=True)
    training_summary.save_plot(
        Path(f"outputs/{cli.run_name}/plots/training_summary.pdf")
    )

    print(
        f"Saving network parameters and hyperparameters in outputs/{cli.run_name}/classifier"
    )

    os.makedirs(f"outputs/{cli.run_name}/classifier", exist_ok=True)
    torch.save(
        network.state_dict(), f"outputs/{cli.run_name}/classifier/parameters.pth"
    )

    save_hyperparameters(
        Path(f"outputs/{cli.run_name}/classifier/hyperparameters.yaml"),
        config.network,
    )

if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig()
    main()
