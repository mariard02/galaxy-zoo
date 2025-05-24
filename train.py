# External imports
import dacite  # For mapping dictionaries to dataclasses
import logging  # Basic logging setup
import os  # OS-level utilities
import shutil  # File copy and move operations
import simple_parsing  # Lightweight CLI parser for dataclasses
import torch  # Core PyTorch library
import yaml  # YAML parsing
import colorful as cf  # For colored terminal output
import pandas as pd

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

def get_class8_weights(label_path: Path) -> torch.Tensor:
    """
    Calcula pesos inversos normalizados para las últimas 14 columnas del CSV.

    Args:
        label_path (Path): Ruta al archivo CSV.

    Returns:
        torch.Tensor: Vector de pesos normalizados.
    """
    df = pd.read_csv(label_path)
    label_cols = df.columns[-14:]  # Toma las últimas 14 columnas
    means = df[label_cols].mean().values
    weights = 1.0 / (means + 1e-6)  # Evita división por cero
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)

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
    label_path = Path("data/exercise_3/train.csv")

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

    galaxy_dataset = load_image_dataset(image_dir, label_path, task = config.network.task_type, transform=None)
    columns = pd.read_csv(label_path, nrows=0).columns.tolist()
    columns.remove("Class2.2")
    #columns.remove("Class6.2")
    columns.remove("GalaxyID")

    print("Preprocessing the data. \n")

    preprocessor = GalaxyPreprocessor(
    image_dir=image_dir,
    label_path=label_path,
    scale_factor=1.0,
    batch_size=config.batch_size,
    normalize=True,
    preprocess_probabilities=False,
    columns=columns,
    n = 3
    )

    galaxy_preprocessed = preprocessor.apply_preprocessing(galaxy_dataset)

    if config.network.task_type == "classification_multiclass":
        weights_binary = GalaxyWeightsClassification(galaxy_dataset)
        weights = weights_binary.get_weights()

        split_dataloader = SplitGalaxyDataLoader(
            galaxy_preprocessed,
            config.validation_fraction,
            config.batch_size,
            class_weights=weights,
            task = config.network.task_type
        )
    else:
        split_dataloader = SplitGalaxyDataLoader(
            galaxy_preprocessed,
            config.validation_fraction,
            config.batch_size,
            class_weights=None,
            task = config.network.task_type
        )

    print("\n Building the CNN. \n")

    #hierarchy_config = {
        #'class1': (None, 2),       # 2 subclases, ninguna padre
        #'class2': ('class1.2', 2), # 2 subclases, hijas de class1.1
        #'class7': ('class1.1', 3)  # 3 subclases, hijas de class1.2
    #}

    hierarchy_config = {
        'class1': (None, 2),  # Class1.1, Class1.2
        
        'class2': ('class1.2', 2),  # Class2.1, Class2.2
        
        'class7': ('class1.1', 3),  # Class7.1, Class7.2, Class7.3
        
        'class6': (None, 2),  # Class6.1, Class6.2
        
        'class8': ('class6.1', 7)  # Class8.1 a Class8.7
    }

    network = build_network(
        galaxy_preprocessed.image_shape(),
        config.network,
        hierarchy_config
    )

    optimizer = AdamW(network.parameters(), lr=config.learning_rate, weight_decay=5.e-5)
    
    weights = get_class8_weights(label_path)
    #loss = WeightedMSELoss(weights=weights)
    #loss = get_loss(config=config)
    loss = HierarchicalFocalLoss(hierarchy_config)

    print_divider()
    print("Training... \n")

    training_summary = galaxy_classification.fit(
        network,
        optimizer,
        loss,
        split_dataloader.training_dataloader,
        split_dataloader.validation_dataloader,
        config.epoch_count,
        patience=20,
        delta = 0.0000001
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