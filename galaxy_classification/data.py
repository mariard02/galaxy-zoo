from dataclasses import dataclass
import torch
from numpy.typing import NDArray
from pathlib import Path
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

# Function to get the label for an image from its filename
def img_label(img, labels_df):
    """
    Given an image path, extract the galaxy ID and retrieve its corresponding label from a DataFrame.

    Parameters:
    img (Path or str): Path object or string representing the image file. 
                       The filename is assumed to be the galaxy ID (e.g., '12345.jpg').

    labels_df (pandas.DataFrame): DataFrame indexed by galaxy IDs, where each row contains 
                                  the labels associated with that ID.

    Returns:
    pandas.Series: The label(s) corresponding to the galaxy ID extracted from the image filename.
    """
    # Extract the image ID from its filename (assuming the filename is the galaxy ID)
    img_id = int(img.name.split('.')[0])  # Use 'name' to get the filename
    
    # If the ID exists, return the corresponding row from labels_df
    return labels_df.loc[img_id]  # Use loc to access the row by its index

# Function to filter a list of files based on whether the filename is in the labels DataFrame index
def trim_file_list(files, labels_df):
    """
    Filter a list of image files to keep only those whose filenames (as IDs) exist in the labels DataFrame.

    Parameters:
    files (list of Path): List of image file paths. Each filename (stem) is assumed to represent a galaxy ID.
    labels_df (pandas.DataFrame): DataFrame indexed by galaxy IDs. Only files with matching IDs will be retained.

    Returns:
    list of Path: Filtered list containing only the files with IDs present in labels_df.
    """
    # Filter the files to keep only those whose ID is present in the labels DataFrame index
    files = [file for file in files if int(file.stem) in labels_df.index]
    return files

# Custom dataset class for loading galaxy images and their corresponding labels
class GalaxyDataset(Dataset):
    def __init__(self, file_list, labels_df, transform=None, task=None):
        """
        Initializes the GalaxyDataset.

        Args:
            file_list (list): List of paths to the image files.
            labels_df (pd.DataFrame): DataFrame containing the labels associated with each image.
            transform (callable, optional): Transformation function to apply to the images 
                                            (e.g., resizing, normalization).
            task (str, optional): Type of task, e.g., "classification_multiclass" or "regression".
        """
        self.file_list = trim_file_list(file_list, labels_df)  # Remove entries without labels
        self.labels_df = labels_df
        self.transform = transform
        self.task = task

    def __getitem__(self, idx):
        """
        Retrieves the image and its label at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple (image_tensor, label_tensor)
        """
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure the image has 3 color channels

        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default to basic ToTensor if no transform is provided
            image = transforms.ToTensor()(image)

        label = self._get_label(img_path)

        # Choose tensor type based on the task
        if self.task == "classification_multiclass":
            label_tensor = torch.tensor(label.values, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label.values, dtype=torch.float32)

        return image, label_tensor

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Number of image-label pairs
        """
        return len(self.file_list)

    def _get_label(self, img_path):
        """
        Internal helper to retrieve the label for a given image path.

        Args:
            img_path (Path): Path to the image file.

        Returns:
            pandas.Series: Label(s) corresponding to the image.
        """
        return img_label(img_path, self.labels_df)

    def image_shape(self):
        """
        Returns the shape of the first transformed image, if possible.

        Returns:
            tuple: Shape of the image tensor (e.g., (3, 224, 224))

        Raises:
            AttributeError: If the image does not have a 'shape' attribute.
        """
        image, _ = self[0]
        try:
            return image.shape
        except AttributeError:
            raise AttributeError(
                "The loaded image does not have a 'shape' attribute. "
                "Make sure your 'transform' function converts the image to a tensor."
            )

        
def load_image_dataset(images_dir: Path, labels_path: Path, task = None, transform=None) -> GalaxyDataset:
    """
    Load galaxy images and their corresponding labels into a custom GalaxyDataset.

    Parameters:
    images_dir (Path): Directory containing the image files (.jpg), where filenames correspond to galaxy IDs.
    labels_path (Path): Path to the CSV file containing the labels. The file must include a 'GalaxyID' column.
    transform (callable, optional): Optional image transformations (e.g., normalization, resizing).

    Returns:
    GalaxyDataset: A PyTorch-compatible dataset containing the filtered image-label pairs.
    """
    # Load the labels into a DataFrame and set 'GalaxyID' as the index for easy lookup
    labels_df = pd.read_csv(labels_path).set_index('GalaxyID')

    # Collect all .jpg image paths from the directory and sort them for consistency
    image_paths = sorted(images_dir.glob("*.jpg"))

    # Raise an error if no images were found
    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {images_dir}")

    # Return the dataset instance using the filtered image paths and labels
    return GalaxyDataset(image_paths, labels_df, task=task, transform=transform)

class GalaxyPreprocessor:
    """
    Enhanced preprocessor class that matches build_transform functionality while maintaining
    additional preprocessing capabilities.
    """
    def __init__(self, image_dir: Path, label_path: Path, scale_factor=1., batch_size=64, normalize=True):
        """
        Initializes the preprocessor with automatic stats computation from the dataset.
        
        Parameters:
        -----------
        image_dir : Path
            Directory containing the images
        label_path : Path
            Path to the labels CSV file
        scale_factor : float
            Factor by which to divide image tensors. Default is 1 (no scaling).
        batch_size : int
            Batch size for computing dataset statistics.
        normalize : bool
            Whether to apply normalization
        """
        self.scale_factor = scale_factor
        self.normalize = normalize
        
        # Calculate base transform for statistics computation
        base_transform = transforms.Compose([
            transforms.Resize((424, 424)),
            transforms.Lambda(lambda x: transforms.functional.crop(x, 180, 180, 64, 64)),
            transforms.ToTensor()
        ])
        
        # Load temporary dataset for stats calculation
        temp_dataset = load_image_dataset(image_dir, label_path, transform=base_transform)
        self.mean, self.std = self.compute_mean_std(temp_dataset, batch_size=batch_size)
        
        # Build the complete transform pipeline
        self.transform = self.build_complete_transform()

    def compute_mean_std(self, dataset, batch_size=64):
        """
        Computes per-channel mean and std for a dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        n_pixels = 0
        sum_ = torch.zeros(3)
        sum_sq = torch.zeros(3)

        for imgs, _ in loader:
            # imgs shape: (B, C, H, W)
            n = imgs.size(0)
            imgs = imgs.view(n, 3, -1)  # Flatten H*W
            sum_ += imgs.sum(dim=(0, 2))
            sum_sq += (imgs ** 2).sum(dim=(0, 2))
            n_pixels += imgs.shape[0] * imgs.shape[2]

        mean = sum_ / n_pixels
        std = (sum_sq / n_pixels - mean ** 2).sqrt()
        return mean.tolist(), std.tolist()

    def build_complete_transform(self):
        transform_list = [
            transforms.Resize((424, 424)),
            transforms.Lambda(lambda x: transforms.functional.crop(x, 180, 180, 64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)

    def apply_preprocessing(self, dataset):
        """
        Applies the complete preprocessing pipeline to a dataset.
        """
        labels_df = dataset.labels_df.copy()

        return GalaxyDataset(
            file_list=dataset.file_list,
            labels_df=labels_df,
            transform=self.transform
        )

    def undo_preprocessing(self, dataset) -> Dataset:
        """
        Reverts scaling, normalization and probability preprocessing.
        Keeps other image transformations.
        """
        base_dataset = dataset
        labels_df = base_dataset.labels_df.copy()


        def inverse_transform(img):
            if hasattr(dataset, 'transform') and dataset.transform:
                img = dataset.transform(img)

            if self.normalize:
                mean = torch.tensor(self.mean).view(-1, 1, 1)
                std = torch.tensor(self.std).view(-1, 1, 1)
                img = img * std + mean

            img = img * self.scale_factor
            return img

        return GalaxyDataset(
            file_list=base_dataset.file_list,
            labels_df=labels_df,
            transform=inverse_transform
        )

@dataclass
class SplitGalaxyDataLoader:
    """
    A helper class to split a dataset into training and validation DataLoaders,
    with optional support for weighted sampling based on class imbalance.

    Attributes:
        training_dataloader (DataLoader): PyTorch DataLoader for the training set.
        validation_dataloader (DataLoader): PyTorch DataLoader for the validation set.
    """

    training_dataloader: DataLoader
    validation_dataloader: DataLoader

    def __init__(
        self,
        dataset: Dataset,
        validation_fraction: float,
        batch_size: int,
        random_seed: int = 38,
        class_weights: list = None,
        task = None
    ):
        """
        Initializes the SplitGalaxyDataLoader by splitting the dataset and optionally
        applying class weighting for the training set.

        Args:
            dataset (Dataset): The complete dataset to split.
            validation_fraction (float): Fraction of data to use for validation.
            batch_size (int): Batch size for both DataLoaders.
            random_seed (int): Seed for reproducibility.
            class_weights (list or Tensor, optional): List or tensor of class weights to handle imbalance.
            task (str, optional): Task type, e.g., "classification_multiclass" or "classification_multilabel".

        Raises:
            ValueError: If dataset doesn't implement __len__, task type is unsupported, 
                        or label format is invalid.
            NotImplementedError: If weighted sampling is requested for multilabel classification.
        """
        if not hasattr(dataset, '__len__'):
            raise ValueError("Input dataset must implement __len__()")

        # Split the dataset
        dataset_length = len(dataset)
        validation_size = int(validation_fraction * dataset_length)
        train_size = dataset_length - validation_size

        training_dataset, validation_dataset = random_split(
            dataset,
            lengths=[train_size, validation_size],
            generator=Generator().manual_seed(random_seed)
        )

        sampler = None  # Default: no sampling

        # Optional: apply weighted sampling if class_weights are provided
        if class_weights is not None:
            all_labels = [sample[1] for sample in training_dataset]

            # Handle different label formats
            if task == "classification_multiclass":
                if all(isinstance(label, (int, float)) for label in all_labels):
                    # Labels are single integer values
                    targets = torch.tensor(all_labels).long()
                elif all(isinstance(label, torch.Tensor) and label.dim() == 1 for label in all_labels):
                    # Labels are one-hot encoded tensors
                    targets = torch.stack(all_labels).argmax(dim=1)
                else:
                    raise ValueError("Unsupported label format for multiclass classification")
            elif task == "classification_multilabel":
                raise NotImplementedError("WeightedRandomSampler not implemented for multilabel classification")
            else:
                raise ValueError(f"Unknown task type: {task}")

            # Compute sample weights and create a sampler
            class_weights = class_weights.clone().detach().float()
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        # Build DataLoaders
        self.training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle if using sampler
            sampler=sampler
        )
        self.validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )


class AutoNormalizeTransform(torch.nn.Module):
    """
    Custom transformation that calculates and applies normalization to tensors.
    Now assumes input is already a tensor.
    """
    def __init__(self, tensor_dataset, batch_size=64):
        super().__init__()
        loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

        mean = 0.
        std = 0.
        nb_samples = 0.

        for data in loader:
            if isinstance(data, (list, tuple)):
                data = data[0] 
            
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        self.normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    def forward(self, x):
        return self.normalize(x)

class GalaxyWeightsClassification:
    """
    Class to compute smoothed class weights for galaxy classification.

    This class combines automatically computed class weights using `compute_class_weight`
    (which balances according to class frequency in the dataset) with uniform weights.
    The `alpha` parameter controls the balance between these two approaches.

    Attributes:
    -----------
    weights : np.ndarray
        Array containing the smoothed weights for each class.

    Methods:
    --------
    get_weights():
        Returns the weights as a PyTorch tensor, ready to be used in loss functions
        like CrossEntropyLoss.
    """

    def __init__(
        self,
        dataset: GalaxyDataset,  # Dataset containing one-hot encoded labels
        alpha=1.0  # Weight of the balanced contribution (1 = fully balanced, 0 = fully uniform)
    ):
        # Convert one-hot encoded labels to class indices (e.g., [0,1,0] -> 1)
        label_classes = np.argmax(dataset.labels_df.values, axis=1)

        # Compute balanced weights based on class frequencies
        balanced_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(label_classes),
            y=label_classes
        )

        # Create uniform weights (equal weight for each class)
        uniform_weights = np.ones_like(balanced_weights)

        # Interpolate between balanced and uniform weights using alpha
        self.weights = alpha * balanced_weights + (1 - alpha) * uniform_weights

    def get_weights(self):
        """
        Returns the computed class weights as a PyTorch tensor.

        Returns:
        --------
        torch.Tensor
            A float32 tensor containing the class weights.
        """
        return torch.tensor(self.weights, dtype=torch.float32)
