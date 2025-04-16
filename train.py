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
    with open(path) as config_file:
        return dacite.from_dict(TrainingConfig, yaml.safe_load(config_file)["training"])
    
def prepare_config(
    output_path: Path, default_path: Path, run_name: str, allow_config_edit: bool
) -> TrainingConfig:
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"Copying {default_path} to {output_path}")
    shutil.copy(default_path, output_path)
    if allow_config_edit:
        _ = input(
            f"Please edit the config in outputs/{run_name}/config.yaml"
            " to set the parameters for this run.\n"
            "After, press enter to continue."
        )
    return load_config(output_path)

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Downsample
    transforms.ToTensor()
])

def save_hyperparameters(path: Path, config: NetworkConfig):
    with open(path, "w") as hyperparameter_cache:
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
    galaxy_dataset = load_image_dataset(Path("data/images/images_training_rev1"), Path("data/exercise_1/labels.csv"), transform=transform)   

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

    optimizer = AdamW(network.parameters(), lr = config.learning_rate) # The optimizer does not depend on the task

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
