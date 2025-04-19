from dataclasses import dataclass
import torch
from numpy.typing import NDArray
from pathlib import Path
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
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
    """
    Custom PyTorch Dataset for loading galaxy images and their corresponding labels.

    Parameters:
    file_list (list of Path): List of image file paths. Filenames (without extensions) are assumed to be galaxy IDs.
    labels_df (pandas.DataFrame): DataFrame indexed by galaxy IDs with corresponding label information.
    transform (callable, optional): Optional transformation to be applied on a PIL image (e.g., torchvision transforms).
    """
    def __init__(self, file_list, labels_df, transform=None):
        # Filter the file list to include only files with IDs present in labels_df
        self.file_list = trim_file_list(file_list, labels_df)
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        # Return the number of valid image files
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieve the image and label at the given index.

        Returns:
        image (Tensor): Transformed image tensor.
        label (Tensor): Corresponding label tensor.
        """
        # Get the image path from the list
        img_path = self.file_list[idx]

        # Open the image and ensure it's in RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Retrieve the corresponding label
        label = self._get_label(img_path)

        return image, torch.tensor(label.values, dtype=torch.long)

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

def load_image_dataset(images_dir: Path, labels_path: Path, transform=None) -> GalaxyDataset:
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
    return GalaxyDataset(image_paths, labels_df, transform=transform)

# TO DO: PREPARE THE PREPROCESSING 
class GalaxyPreprocessor:
    """
    Class to preprocess galaxy images by applying a scaling transformation.
    
    The preprocessing can be undone by reversing the scaling transformation.
    """
    def __init__(self, scale_factor=1):
        """
        Initialize the GalaxyPreprocessor with a scaling factor.
        
        :param scale_factor: A float representing the factor by which to scale the images. 
                              Default is 256.0.
        """
        self.scale_factor = scale_factor

    def apply_preprocessing(self, dataset: GalaxyDataset) -> GalaxyDataset:
        """
        Apply preprocessing to the dataset, which includes the scaling transformation.
        
        This method creates a new transformation function that applies the existing 
        transformation (if any) followed by scaling the images by the defined scale factor.
        
        :param dataset: The GalaxyDataset to be preprocessed.
        :return: A new GalaxyDataset with the preprocessing transformation applied.
        """
        # Define a new transformation function that first applies any existing transformation
        # and then scales the image.
        def new_transform(img):
            if dataset.transform:
                img = dataset.transform(img)
            return img / self.scale_factor
        
        # Return a new dataset with the new transformation
        return GalaxyDataset(
            file_list=dataset.file_list,
            labels_df=dataset.labels_df,
            transform=new_transform
        )

    def undo_preprocessing(self, dataset: GalaxyDataset) -> GalaxyDataset:
        """
        Undo the preprocessing (i.e., the scaling) applied earlier to the dataset.
        
        This method creates a new transformation function that reverses the scaling transformation
        by multiplying the image by the scale factor instead of dividing.
        
        :param dataset: The GalaxyDataset from which to undo the preprocessing.
        :return: A new GalaxyDataset with the inverse transformation applied.
        """
        # Define a new transformation function that applies the inverse of the scaling transformation
        def new_transform(img):
            if dataset.transform:
                img = dataset.transform(img)
            return img * self.scale_factor

        # Return a new dataset with the inverse transformation
        return GalaxyDataset(
            file_list=dataset.file_list,
            labels_df=dataset.labels_df,
            transform=new_transform
        )

@dataclass
class SplitGalaxyDataLoader:
    """
    Data loader class for splitting a GalaxyDataset into training and validation sets.
    
    The dataset is divided based on a specified validation fraction, and DataLoader objects are
    created for both the training and validation datasets.
    """
    training_dataloader: DataLoader
    validation_dataloader: DataLoader

    def __init__(
        self,
        dataset: GalaxyDataset,  # GalaxyDataset used as input
        validation_fraction: float,  # Fraction of data to be used for validation
        batch_size: int,  # Batch size for the data loaders
    ):
        """
        Initialize the data loaders for training and validation datasets.
        
        The dataset is split based on the provided validation fraction, and 
        DataLoader objects are created for both training and validation sets.
        
        :param dataset: GalaxyDataset containing the galaxy images and their labels.
        :param validation_fraction: A float representing the fraction of the dataset to be used 
                                    for validation.
        :param batch_size: The batch size to be used in the DataLoader objects.
        """
        # Calculate the number of samples for training and validation sets
        validation_size = int(validation_fraction * len(dataset))
        train_size = len(dataset) - validation_size

        # Split the dataset into training and validation sets
        training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, validation_size],
            generator=Generator().manual_seed(20),  # Ensuring reproducibility of the split
        )

        # Create DataLoader objects for both training and validation datasets
        self.training_dataloader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True
        )

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
