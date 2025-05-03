# External imports
import dacite  # For mapping dictionaries to dataclasses
import logging  # Basic logging setup
import os  # OS-level utilities
import shutil  # File copy and move operations
import simple_parsing  # Lightweight CLI parser for dataclasses
import torch  # Core PyTorch library
import yaml  # YAML parsing
import colorful as cf  # For colored terminal output

# Standard library tools for structure and typing
from dataclasses import asdict, dataclass
from pathlib import Path

# PyTorch modules
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Local imports (from your project)
from galaxy_classification.data import *
from galaxy_classification.networks import *
import galaxy_classification
from asciiart import *

# Set a terminal color style for better visual feedback
cf.use_style('monokai')


# === Dataclasses ===

@dataclass
class TrainingCli:
    """
    Command-line arguments for a training run.

    Attributes:
        run_name (str): Unique identifier for the training run (used in filenames/paths).
        no_config_edit (bool): If True, skip prompting user to edit the config file before running.
    """
    run_name: str
    no_config_edit: bool = False

@dataclass
class TrainingConfig:
    """
    Configuration structure for training, loaded from YAML.

    Attributes:
        epoch_count (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        validation_fraction (float): Fraction of the dataset used for validation.
        network (NetworkConfig): Nested config for the network architecture.
    """
    epoch_count: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    network: NetworkConfig


# === Config Utilities ===

def load_config(path: Path) -> TrainingConfig:
    """
    Load training configuration from a YAML file.

    Args:
        path (Path): Path to YAML config.

    Returns:
        TrainingConfig: Parsed configuration object.
    """
    with open(path) as config_file:
        return dacite.from_dict(TrainingConfig, yaml.safe_load(config_file)["training"])

def prepare_config(output_path: Path, default_path: Path, run_name: str, allow_config_edit: bool) -> TrainingConfig:
    """
    Prepare a training configuration file, with optional user editing.

    Args:
        output_path (Path): Path where run-specific config will be written.
        default_path (Path): Path to default config to copy.
        run_name (str): Name of current run.
        allow_config_edit (bool): If True, prompts user to manually edit the config.

    Returns:
        TrainingConfig: Loaded and parsed config.
    """
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"Copying {default_path} to {output_path}")
    shutil.copy(default_path, output_path)

    if allow_config_edit:
        _ = input(
            f"Please edit the config in outputs/{run_name}/config.yaml\n"
            "Then press Enter to continue."
        )

    return load_config(output_path)


# === Data Transformations ===

def build_transform(image_dir: Path, label_path: Path) -> torch.nn.Module:
    """
    Build transform pipeline for dataset preprocessing.

    Includes resizing, cropping, flips, conversion to tensor, and normalization.

    Args:
        image_dir (Path): Path to image data.
        label_path (Path): Path to label CSV.

    Returns:
        torch.nn.Module: Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((424, 424)),
        transforms.Lambda(lambda x: transforms.functional.crop(x, 180, 180, 64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        AutoNormalizeTransform(image_dir, label_path),
    ])

# Additional transformation for data augmentation
transform_1_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
])


# === Hyperparameter Saving ===

def save_hyperparameters(path: Path, config: NetworkConfig):
    """
    Save network hyperparameters to a YAML file.

    Args:
        path (Path): Output path for YAML file.
        config (NetworkConfig): Network configuration dataclass.
    """
    with open(path, "w") as hyperparameter_cache:
        yaml.dump(asdict(config), hyperparameter_cache)


# === Main Training Logic ===

def main():
    """
    Main training pipeline.

    - Parses CLI arguments.
    - Loads dataset and config.
    - Applies preprocessing.
    - Initializes model, optimizer, loss.
    - Trains and saves results.
    """
    parser = simple_parsing.ArgumentParser(add_help=True, description="Train the galaxy classifier.")
    parser.add_arguments(TrainingCli, dest="cli")
    args = parser.parse_args()
    cli: TrainingCli = args.cli

    image_dir = Path("data/images/images_training_rev1")
    label_path = Path("data/exercise_1/labels.csv")

    print("\n" + cf.purple(generate_title_string()) + "\n") 
    print_divider()
    print(f"Run name: {cf.purple(cli.run_name)}\n")

    config = prepare_config(
        Path(f"outputs/{cli.run_name}/config.yaml"),
        Path("config_default.yaml"),
        cli.run_name,
        not cli.no_config_edit,
    )

    print("\nLoading the dataset. \n")

    transform = build_transform(image_dir=image_dir, label_path=label_path)
    galaxy_dataset = load_image_dataset(image_dir, label_path, task = config.network.task_type, transform=transform)

    galaxy_dataset = load_custom_image_dataset(galaxy_dataset, None, transform_1_3)

    print("Preprocessing the data. \n")
    preprocessor = GalaxyPreprocessor()
    galaxy_preprocessed = preprocessor.apply_preprocessing(galaxy_dataset)

    weights_binary = GalaxyWeightsClassification(galaxy_dataset)
    weights = weights_binary.get_weights()

    split_dataloader = SplitGalaxyDataLoader(
        galaxy_preprocessed,
        config.validation_fraction,
        config.batch_size,
        class_weights=weights,
        task = config.network.task_type
    )

    print("Building the CNN. \n")
    network = build_network(
        galaxy_preprocessed.image_shape(),
        config.network,
    )

    optimizer = AdamW(network.parameters(), lr=config.learning_rate, weight_decay=5.e-3)
    
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
        patience=5,
        delta = 0.01
    )

    print(f"Saving training summary plots to outputs/{cli.run_name}/plots/training_summary.pdf")
    os.makedirs(f"outputs/{cli.run_name}/plots", exist_ok=True)
    training_summary.save_plot(Path(f"outputs/{cli.run_name}/plots/training_summary.pdf"))

    print(f"Saving network parameters and hyperparameters in outputs/{cli.run_name}/classifier")
    os.makedirs(f"outputs/{cli.run_name}/classifier", exist_ok=True)
    torch.save(network.state_dict(), f"outputs/{cli.run_name}/classifier/parameters.pth")

    save_hyperparameters(
        Path(f"outputs/{cli.run_name}/classifier/hyperparameters.yaml"),
        config.network,
    )

# Run the script
if __name__ == "__main__":
    logging.basicConfig()
    main()