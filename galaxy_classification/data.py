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

# Dataset class for loading galaxy images
class GalaxyDataset(Dataset):
    def __init__(self, file_list, labels_df, transform=None, task=None):
        self.file_list = trim_file_list(file_list, labels_df)
        self.labels_df = labels_df
        self.transform = transform
        self.task = task  # Añadido

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self._get_label(img_path)

        # Cambio de tipo en función de la tarea
        if self.task == "classification_multiclass":
            label_tensor = torch.tensor(label.values, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label.values, dtype=torch.float32)

        return image, label_tensor

    def __len__(self):
        # Return the number of valid image files
        return len(self.file_list)

    def _get_label(self, img_path):
        """
        Internal helper to retrieve the label for a given image path.

        Parameters:
        img_path (Path): Path to the image file.

        Returns:
        pandas.Series: Label(s) corresponding to the image.
        """
        return img_label(img_path, self.labels_df)
    
    def image_shape(self):
        """
        Returns the shape of the first transformed image, if possible.

        Returns:
            tuple: Shape of the image (e.g., (3, 224, 224))

        Raises:
            AttributeError: If the image does not have a 'shape' attribute.
        """
        image, _ = self[0]
        try:
            return image.shape
        except AttributeError:
            raise AttributeError("The loaded image does not have a 'shape' attribute. "
                                "Make sure your 'transform' function converts the image to a tensor.")
        
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
        
class CustomAugmentedDataset(Dataset):
    """
    A wrapper dataset that applies a special transformation to a specific class
    while applying a standard transformation to all other samples.

    Attributes:
    -----------
    dataset : Dataset
        The base dataset containing images and labels.
    transform_normal : callable
        Standard transformation applied to all images.
    transform_1_3 : callable
        Special transformation applied only to class 1.3 samples.
    """
    def __init__(self, dataset, transform_normal, transform_1_3):
        self.dataset = dataset
        self.transform_normal = self._ensure_tensor_transform(transform_normal)
        self.transform_1_3 = self._ensure_tensor_transform(transform_1_3)
        self.file_list = getattr(dataset, 'file_list', None)
        self.labels_df = getattr(dataset, 'labels_df', None)

    def _ensure_tensor_transform(self, transform):
        """
        Ensures the output of the transform is always a tensor.
        Useful in case the input transform does not convert to tensor.
        """
        if transform is None:
            return transforms.ToTensor()
        
        return transforms.Compose([
            transform,
            transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x))
        ])

    def __getitem__(self, idx):
        """
        Returns a transformed image and its label.
        Applies `transform_1_3` if the sample belongs to class 1.3,
        otherwise applies `transform_normal`.

        Parameters:
        -----------
        idx : int
            Index of the sample to retrieve.

        Returns:
        --------
        (img, label) : tuple
            Transformed image and its corresponding label.
        """
        img, label = self.dataset[idx]
        
        # Apply the normal transform first (guaranteed to return a tensor)
        #img = self.transform_normal(img)

        # Decide whether to apply the special class 1.3 transform
        if isinstance(label, torch.Tensor):
            if label.dim() > 0:  
                if label[2] == 1:
                    img = self.transform_1_3(img)

        return img, label

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def image_shape(self):
        """
        Returns the shape of a transformed image.
        Useful to verify dimensions before defining a model.
        """
        try:
            img, _ = self[0]
            return img.shape
        except Exception as e:
            raise RuntimeError(f"Could not determine image shape: {str(e)}")


def load_custom_image_dataset(dataset: GalaxyDataset, transform=None, transform2=None) -> GalaxyDataset:
    """
    Wraps a GalaxyDataset with custom transformations using CustomAugmentedDataset.

    Parameters:
    -----------
    dataset : GalaxyDataset
        The base dataset containing image-label pairs.
    transform : callable, optional
        The standard transformation to apply to all images.
    transform2 : callable, optional
        The special transformation to apply only to class 1.3 images.

    Returns:
    --------
    CustomAugmentedDataset
        A dataset that applies different transforms depending on the label.
    """
    return CustomAugmentedDataset(dataset=dataset, transform_normal=transform, transform_1_3=transform2)

class GalaxyPreprocessor:
    """
    Class to apply and undo preprocessing transformations on Galaxy datasets,
    including scaling and normalization.
    """
    def __init__(self, dataset, scale_factor=1., mean=None, std=None, batch_size=64, normalize=False):
        """
        Initializes the preprocessor with a scale factor and optional normalization.
        If mean and std are not provided, they will be computed from the dataset.

        Parameters:
        -----------
        dataset : GalaxyDataset or CustomAugmentedDataset
            Dataset to use for computing statistics if needed.
        scale_factor : float
            Factor by which to divide image tensors. Default is 1 (no scaling).
        mean : list or None
            Channel-wise means. If None, they will be computed.
        std : list or None
            Channel-wise stds. If None, they will be computed.
        batch_size : int
            Batch size for computing dataset statistics.
        """
        self.scale_factor = scale_factor
        self.mean = mean
        self.std = std
        self.normalize = normalize

        if self.mean is None or self.std is None:
            self.mean, self.std = self.compute_mean_std(dataset, batch_size=batch_size)

    def compute_mean_std(self, dataset, batch_size=64):
        """
        Computes per-channel mean and std for a dataset.
        """
        # Use a basic transform to ensure tensors are produced
        def to_tensor(img):
            return transforms.ToTensor()(img) / self.scale_factor

        base_dataset = dataset.dataset if isinstance(dataset, CustomAugmentedDataset) else dataset
        temp_dataset = GalaxyDataset(
            file_list=base_dataset.file_list,
            labels_df=base_dataset.labels_df,
            transform=to_tensor
        )

        loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        n_pixels = 0
        sum_ = torch.zeros(3)
        sum_sq = torch.zeros(3)

        for imgs, _ in tqdm(loader):
            # imgs shape: (B, C, H, W)
            n = imgs.size(0)
            imgs = imgs.view(n, 3, -1)  # Flatten H*W
            sum_ += imgs.sum(dim=(0, 2))
            sum_sq += (imgs ** 2).sum(dim=(0, 2))
            n_pixels += imgs.shape[0] * imgs.shape[2]

        mean = sum_ / n_pixels
        std = (sum_sq / n_pixels - mean ** 2).sqrt()
        return mean.tolist(), std.tolist()

    def apply_preprocessing(self, dataset):
        """
        Applies scaling and normalization to the dataset images.
        """
        def new_transform(img):
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
                if self.normalize:
                    img = img / self.scale_factor
                    img = transforms.Normalize(self.mean, self.std)(img)
            return img

        if isinstance(dataset, CustomAugmentedDataset):
            return CustomAugmentedDataset(
                dataset=dataset.dataset,
                transform_normal=new_transform,
                transform_1_3=dataset.transform_1_3
            )
        else:
            return GalaxyDataset(
                file_list=dataset.file_list,
                labels_df=dataset.labels_df,
                transform=new_transform
            )

    def undo_preprocessing(self, dataset) -> Dataset:
        """
        Reverts scaling and normalization.
        """
        base_dataset = dataset.dataset if isinstance(dataset, CustomAugmentedDataset) else dataset

        def new_transform(img):
            if isinstance(dataset, CustomAugmentedDataset):
                img = dataset.transform_normal(img)
            elif hasattr(dataset, 'transform') and dataset.transform:
                img = dataset.transform(img)

            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            img = img * std + mean
            img = img * self.scale_factor
            return img

        if isinstance(dataset, CustomAugmentedDataset):
            return CustomAugmentedDataset(
                dataset=dataset.dataset,
                transform_normal=new_transform,
                transform_1_3=dataset.transform_1_3
            )
        else:
            return GalaxyDataset(
                file_list=base_dataset.file_list,
                labels_df=base_dataset.labels_df,
                transform=new_transform
            )


@dataclass
class SplitGalaxyDataLoader:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader

    def __init__(
        self,
        dataset: Dataset,
        validation_fraction: float,
        batch_size: int,
        random_seed: int = 38,
        class_weights: list = None,  # Optional class weights
        task = None
    ):
        if not hasattr(dataset, '__len__'):
            raise ValueError("Input dataset must implement __len__()")

        dataset_length = len(dataset)
        validation_size = int(validation_fraction * dataset_length)
        train_size = dataset_length - validation_size

        training_dataset, validation_dataset = random_split(
            dataset,
            lengths=[train_size, validation_size],
            generator=Generator().manual_seed(random_seed)
        )

        sampler = None

        if class_weights is not None:
            all_labels = [sample[1] for sample in training_dataset]
            
            # Handle different label formats
            if task == "classification_multiclass":
                # For multiclass classification with single integer labels
                if all(isinstance(label, (int, float)) for label in all_labels):
                    targets = torch.stack(all_labels).long()
                # For multiclass classification with one-hot encoded labels
                elif all(isinstance(label, torch.Tensor) and label.dim() == 1 for label in all_labels):
                    targets = torch.stack(all_labels).argmax(dim=1)
                else:
                    raise ValueError("Unsupported label format for multiclass classification")
            elif task == "classification_multilabel":
                # For multilabel classification, WeightedRandomSampler isn't directly applicable
                # You might need a different approach here
                raise NotImplementedError("WeightedRandomSampler not implemented for multilabel classification")
            else:
                raise ValueError(f"Unknown task type: {task}")
            
            class_weights = class_weights.clone().detach().float()
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        self.training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=False,  
            sampler=sampler
        )
        self.validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

class AutoNormalizeTransform(torch.nn.Module):
    """
    Custom transformation module that automatically calculates and applies 
    normalization to images based on their dataset. This module computes 
    the mean and standard deviation of the dataset and uses these values 
    to normalize the images during the forward pass.

    Args:
        image_dir (Path): The directory where the image files are stored.
        label_path (Path): The path to the CSV file containing the labels.
        image_size (tuple, optional): Desired image size after resizing. Default is (32, 32).
        batch_size (int, optional): The batch size used for calculating statistics. Default is 64.
    """
    
    def __init__(self, image_dir: Path, label_path: Path, image_size=(32, 32), batch_size=64):
        super().__init__()

        # Base transformation for resizing and converting images to tensors.
        # This is used only for calculating the statistics (mean and std).
        base_transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize images to the specified size
            transforms.ToTensor()           # Convert images to Tensor format (with values in [0, 1])
        ])
        
        # Create a temporary dataset using the provided image and label paths, applying the base transformation.
        temp_dataset = load_image_dataset(image_dir, label_path, transform=base_transform)
        
        # DataLoader is used to load the dataset in batches without shuffling.
        loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

        # Initialize variables to accumulate the mean, std, and number of samples.
        mean = 0.
        std = 0.
        nb_samples = 0.

        # Iterate through the DataLoader to calculate the mean and std for the dataset.
        for data, _ in loader:
            batch_samples = data.size(0)  # Get the batch size
            data = data.view(batch_samples, data.size(1), -1)  # Flatten the images to 2D for mean/std computation

            # Sum the means and stds across the batch (per channel).
            mean += data.mean(2).sum(0)  # Compute the mean for each channel
            std += data.std(2).sum(0)    # Compute the standard deviation for each channel
            nb_samples += batch_samples  # Count the number of samples processed

        # Calculate the final mean and std by averaging over all samples.
        mean /= nb_samples
        std /= nb_samples

        # Create a normalization transform using the calculated mean and std.
        self.normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    def forward(self, x):
        """
        Forward pass for normalizing the input image tensor.
        
        Args:
            x (Tensor): Input image tensor to be normalized.
        
        Returns:
            Tensor: Normalized image tensor.
        """
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
