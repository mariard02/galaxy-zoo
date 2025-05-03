from dataclasses import dataclass
from pathlib import Path
from typing import get_args
import dacite
from pydantic import BaseModel
import simple_parsing
import torch
from torch.utils.data import DataLoader
import yaml

from galaxy_classification import *
from galaxy_classification.data import *
from galaxy_classification.networks import *

import matplotlib.pyplot as plt

# Define a dataclass to handle command-line input for evaluation run name.
@dataclass
class EvaluationCli:
    run_name: str


# Configuration class for evaluation parameters, including batch size and task type (classification type).
@dataclass
class EvaluationConfig(BaseModel):
    batch_size: int
    task_type: str


# Load configuration settings from a YAML file into an EvaluationConfig object.
def load_config(path: Path) -> EvaluationConfig:
    with open(path) as config_file:
        # Load the YAML file and validate the structure using the Pydantic model
        return EvaluationConfig.model_validate(
            yaml.safe_load(config_file)["evaluation"]
        )


# Load hyperparameters for the network from a YAML file and convert them into the appropriate configuration object.
def load_hyperparameters(path: Path) -> GalaxyClassificationCNNConfig:
    with open(path) as hyperparameter_cache:
        network_config = yaml.safe_load(hyperparameter_cache)
        # Use dacite to map the dictionary to a GalaxyClassificationCNNConfig object
        return dacite.from_dict(GalaxyClassificationCNNConfig, network_config)

# Function to build the image preprocessing pipeline (transformations).
def build_transform(image_dir: Path, label_path: Path) -> torch.nn.Module:
    """
    Create a series of transformations to preprocess the dataset, including resizing, cropping,
    random flips, rotation, and normalization.

    Args:
        image_dir (Path): Path to the directory containing image data.
        label_path (Path): Path to the CSV file with labels.

    Returns:
        torch.nn.Module: A composed transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((424, 424)),
        transforms.Lambda(lambda x: transforms.functional.crop(x, 180, 180, 64, 64)),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip
        transforms.RandomRotation(30),  # Random rotation by 30 degrees
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        AutoNormalizeTransform(image_dir, label_path),  # Custom normalization based on the image and label paths
    ])

# Custom set of transformations (flip, rotate)
transform_1_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
])

# Main function to load data, preprocess it, and evaluate the model.
def main():
    cli = simple_parsing.parse(EvaluationCli)  # Parse CLI arguments to get run name
    config = load_config(Path(f"outputs/{cli.run_name}/config.yaml"))  # Load the evaluation config

    # Set paths for image data and labels
    image_dir = Path("data/images/images_training_rev1")
    label_path = Path("data/exercise_1/labels.csv")

    print("loading the dataset")

    # Preprocess dataset using the defined transformations
    print("preprocessing the dataset")
    
    transform = build_transform(image_dir=image_dir, label_path=label_path)
    galaxy_dataset = load_image_dataset(image_dir, label_path, task=config.task_type, transform=transform)
    galaxy_dataset = load_custom_image_dataset(galaxy_dataset, None, transform_1_3)
    preprocessor = GalaxyPreprocessor(dataset=galaxy_dataset, batch_size=config.batch_size, normalize=True)
    galaxy_preprocessed = preprocessor.apply_preprocessing(galaxy_dataset)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(
        galaxy_preprocessed, batch_size=config.batch_size, shuffle=False  # Important to set shuffle=False for ROC
    )

    # Load network hyperparameters and model configuration
    network_config = load_hyperparameters(
        Path(f"outputs/{cli.run_name}/classifier/hyperparameters.yaml")
    )
    network = build_network(galaxy_preprocessed.image_shape(), network_config)
    network.load_state_dict(
        torch.load(f"outputs/{cli.run_name}/classifier/parameters.pth")
    )

    network.eval()  # Set the network to evaluation mode

    # Compute accuracy on the validation/test set
    print("computing the accuracy")
    accuracy = compute_accuracy(network, dataloader, config.task_type)
    print(f"accuracy = {accuracy:.2f}")

    # Collect predictions and labels for ROC curve plotting
    print("computing ROC curves")
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in dataloader:
            outputs = network(images)
            if config.task_type == "classification_multiclass":
                probs = torch.softmax(outputs, dim=1)
            elif config.task_type == "classification_binary":
                probs = torch.sigmoid(outputs).unsqueeze(1)
            else:
                raise ValueError(f"Unknown task type {config.task_type}")
            all_preds.append(probs)
            all_labels.append(labels)

    # Concatenate all predictions and labels into tensors
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    if all_labels.ndim > 1 and config.task_type == "classification_multiclass":
        all_labels = torch.argmax(all_labels, dim=1)  # Convert one-hot to class indices if needed
    
    class_names = ["Smooth", "Disk", "Other"]

    # Plot the ROC curves and save the plot
    plot_roc_curves(all_preds, all_labels, config, Path(f"outputs/{cli.run_name}/plots/ROC_curve.pdf"), class_names)

    # Plot the confussion matrix and save the plot
    plot_confusion_matrix(all_preds, all_labels, config, Path(f"outputs/{cli.run_name}/plots/confussion_matrix.pdf"), class_names)

# Run the main function
if __name__ == "__main__":
    main()
