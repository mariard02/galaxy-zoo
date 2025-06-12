from dataclasses import dataclass
from pathlib import Path
from typing import get_args, Optional
import dacite
from pydantic import BaseModel
import simple_parsing
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from galaxy_classification import *
from galaxy_classification.data import *
from galaxy_classification.networks import *

import matplotlib.pyplot as plt

@dataclass
class EvaluationCli:
    run_name: str

@dataclass
class EvaluationConfig(BaseModel):
    batch_size: int
    task_type: str  # "classification_multiclass" or "regression"

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
    return transforms.Compose([
        transforms.Resize((424, 424)),
        transforms.Lambda(lambda x: transforms.functional.crop(x, 180, 180, 64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        AutoNormalizeTransform(image_dir, label_path),
    ])

def load_hierarchy_config(path: Path) -> dict:
    """
    Load hierarchy configuration from YAML or JSON file.
    
    Args:
        path (Path): Path to hierarchy config file.
        
    Returns:
        dict: Parsed hierarchy configuration.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config["hierarchy"]

def compute_regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Compute regression metrics (MSE, MAE, R²)"""
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    
    return {
        "mse": mean_squared_error(y_true_np, y_pred_np),
        "mae": mean_absolute_error(y_true_np, y_pred_np),
        "r2": r2_score(y_true_np, y_pred_np)
    }

def main():
    cli = simple_parsing.parse(EvaluationCli)
    config = load_config(Path(f"outputs/{cli.run_name}/config.yaml"))

    image_dir = Path("data/images/images_training_rev1")
    label_path = Path("data/exercise_1/test.csv")

    if config.task_type == "regression":
        hierarchy_config = load_hierarchy_config(Path("data/exercise_2/hierarchy.yaml"))
        hierarchy_config = {
            class_name: (info["parent"], info["num_classes"])
            for class_name, info in hierarchy_config.items()
        }
    else: 
        hierarchy_config = None

    print("\nLoading the dataset. \n")

    galaxy_dataset = load_image_dataset(image_dir, label_path, None, transform=None)

    print("Preprocessing the data. \n")

    preprocessor = GalaxyPreprocessor(
    image_dir=image_dir,
    label_path=label_path,
    scale_factor=1.0,
    batch_size=config.batch_size,
    normalize=True,
    )

    galaxy_preprocessed = preprocessor.apply_preprocessing(galaxy_dataset)

    dataloader = DataLoader(
        galaxy_preprocessed, batch_size=config.batch_size, shuffle=False
    )

    hierarchy_config = {
        'class1': (None, 3),      
        'class2': ('class1.2', 2), 
        'class7': ('class1.1', 3) 
    }

    network_config = load_hyperparameters(
        Path(f"outputs/{cli.run_name}/classifier/hyperparameters.yaml")
    )# Before loading the model
    network = build_network(galaxy_preprocessed.image_shape(), network_config, hierarchy_config=hierarchy_config)

    # Verify this matches the architecture used during training
    network.load_state_dict(
        torch.load(f"outputs/{cli.run_name}/classifier/parameters.pth"),
    )   
    network.eval()

    # For all tasks
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = network(images)
            
            if config.task_type == "classification_multiclass":
                probs = torch.softmax(outputs, dim=1)
            else:  # regression
                probs = outputs  # Direct predictions
                
            all_preds.append(probs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Task-specific evaluation
    if config.task_type == "regression":

        # Compute regression metrics
        metrics = compute_regression_metrics(all_labels, all_preds)
        plot_feature_histograms(all_preds, all_labels, [0., 1.], Path(f"outputs/{cli.run_name}/plots/histograms.pdf"))
        print("Regression Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        
    else:  # Classification tasks
        # Compute accuracy
        accuracy = compute_accuracy(network, dataloader, config.task_type)
        print(f"accuracy = {accuracy:.2f}")

        # For multiclass classification, prepare class names and plots
        if config.task_type == "classification_multiclass":
            if all_labels.ndim > 1:
                all_labels = torch.argmax(all_labels, dim=1)
            
            class_names = ["Smooth", "Disk"]
            
            # Plot ROC curves and confusion matrix
            plot_roc_curves(
                all_preds, 
                all_labels, 
                config, 
                Path(f"outputs/{cli.run_name}/plots/ROC_curve.pdf"), 
                class_names
            )
            
            plot_confusion_matrix(
                all_preds, 
                all_labels, 
                config, 
                Path(f"outputs/{cli.run_name}/plots/confusion_matrix.pdf"), 
                class_names
            )

if __name__ == "__main__":
    main()