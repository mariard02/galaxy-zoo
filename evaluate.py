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


@dataclass
class EvaluationCli:
    run_name: str


@dataclass
class EvaluationConfig(BaseModel):
    batch_size: int
    task_type: str


def load_config(path: Path) -> EvaluationConfig:
    with open(path) as config_file:
        return EvaluationConfig.model_validate(
            yaml.safe_load(config_file)["evaluation"]
        )


def load_hyperparameters(path: Path) -> GalaxyClassificationCNNConfig:
    with open(path) as hyperparameter_cache:
        network_config = yaml.safe_load(hyperparameter_cache)
        return dacite.from_dict(GalaxyClassificationCNNConfig, network_config)

    

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
        transforms.Resize((128, 128)),
        transforms.Lambda(lambda x: transforms.functional.crop(x, 32, 32, 64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        AutoNormalizeTransform(image_dir, label_path),
    ])

transform_1_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
])

def main():
    cli = simple_parsing.parse(EvaluationCli)
    config = load_config(Path(f"outputs/{cli.run_name}/config.yaml"))

    image_dir = Path("data/images/images_training_rev1")
    label_path = Path("data/exercise_1/labels.csv")

    print("loading the dataset")

    print("preprocessing the dataset")
    preprocessor = GalaxyPreprocessor()
    transform = build_transform(image_dir=image_dir, label_path=label_path)
    galaxy_dataset = load_image_dataset(image_dir, label_path, task=config.task_type, transform=transform)
    galaxy_dataset = load_custom_image_dataset(galaxy_dataset,transform, transform_1_3)
    galaxy_preprocessed = preprocessor.apply_preprocessing(galaxy_dataset)

    dataloader = DataLoader(
        galaxy_preprocessed, batch_size=config.batch_size, shuffle=True
    )

    network_config = load_hyperparameters(
        Path(f"outputs/{cli.run_name}/classifier/hyperparameters.yaml")
    )
    network = build_network(galaxy_preprocessed.image_shape(), network_config)
    network.load_state_dict(
        torch.load(f"outputs/{cli.run_name}/classifier/parameters.pth")
    )

    print("computing the accuracy")
    accuracy = compute_accuracy(network, dataloader, config.task_type)
    print(f"accuracy = {accuracy:.2f}")


if __name__ == "__main__":
    main()
