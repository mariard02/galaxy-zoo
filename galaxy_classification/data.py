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
        label = self.__get_label(img_path)

        return image, torch.tensor(label.values, dtype=torch.float64)

    def __get_label(self, img_path):
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

class GalaxyPreprocessor:
    def __init__(self, scale_factor=256.0):
        self.scale_factor = scale_factor

    def apply_preprocessing(self, dataset: GalaxyDataset) -> GalaxyDataset:
        # Creamos un nuevo transform que aplica el anterior + el escalado
        def new_transform(img):
            if dataset.transform:
                img = dataset.transform(img)
            return img / self.scale_factor
        
        return GalaxyDataset(
            file_list=dataset.file_list,
            labels_df=dataset.labels_df,
            transform=new_transform
        )

    def undo_preprocessing(self, dataset: GalaxyDataset) -> GalaxyDataset:
        def new_transform(img):
            if dataset.transform:
                img = dataset.transform(img)
            return img * self.scale_factor

        return GalaxyDataset(
            file_list=dataset.file_list,
            labels_df=dataset.labels_df,
            transform=new_transform
        )

@dataclass
class SplitGalaxyDataLoader:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader

    def __init__(
        self,
        dataset: GalaxyDataset,  # Usamos GalaxyDataset
        validation_fraction: float,
        batch_size: int,
    ):
        validation_size = int(validation_fraction * len(dataset))
        train_size = len(dataset) - validation_size

        # Dividir el dataset en entrenamiento y validación
        training_dataset, validation_dataset = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, validation_size],
            generator=Generator().manual_seed(42),
        )

        # Crear los dataloaders
        self.training_dataloader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True
        )