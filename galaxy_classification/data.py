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
    '''Function to turn image path into ID, and return labels from labels_df'''
    # Extract the image ID from its filename (assuming the filename is the galaxy ID)
    img_id = int(img.name.split('.')[0])  # Use 'name' to get the filename
    
    # Check if the ID is in the index of labels_df
    if img_id not in labels_df.index:
        print(f"Warning: Image ID {img_id} not found in labels DataFrame. Skipping this image.")
        return None  # Return None if the ID is not found in the labels DataFrame
    
    # If the ID exists, return the corresponding row from labels_df
    return labels_df.loc[img_id]  # Use loc to access the row by its index

# Function to filter a list of files based on whether the filename is in the labels DataFrame index
def trim_file_list(files, labels_df):
    '''Function to trim a list of files based on whether the file name is in the ID of labels_df'''
    # Filter the files to keep only those whose ID is present in the labels DataFrame index
    files = [file for file in files if int(file.stem) in labels_df.index]
    print(f"Filtered file list length: {len(files)}")  # Debugging line
    return files


# Dataset class for loading galaxy images
class GalaxyDataset(Dataset):
    def __init__(self, file_list, labels_df, transform=None):
        # Filter the file list to keep only files that match the IDs in labels_df
        self.file_list = trim_file_list(file_list, labels_df)
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the image path for the given index
        img_path = self.file_list[idx]
        # Open the image and convert it to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Get the label for the image
        label = self._get_label(img_path)
        return image, torch.tensor(label, dtype=torch.long)

    def _get_label(self, img_path):
        '''Extracts the label for the given image path'''
        # Use the img_label function to get the label for the image
        return img_label(img_path, self.labels_df)


# Function to load the image dataset
def load_image_dataset(images_dir: Path, labels_path: Path, transform=None) -> GalaxyDataset:
    # Load the labels into a DataFrame and set 'GalaxyID' as the index
    labels_df = pd.read_csv(labels_path).set_index('GalaxyID')

    # Get all the image paths (sorted list of .jpg files)
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {images_dir}")

    # If the list is empty, there's an issue with the file search pattern
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {images_dir}")

    # Return a GalaxyDataset instance
    return GalaxyDataset(image_paths, labels_df, transform=transform)

# Define any transformations you want to apply to the images (e.g., resizing and converting to tensor)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = load_image_dataset(
    images_dir=Path("/Users/maria/Desktop/Máster/M1/S2/Physics applications of AI/galaxy-zoo/galaxy-zoo/data/images/images_training_rev1"),
    labels_path=Path("/Users/maria/Desktop/Máster/M1/S2/Physics applications of AI/galaxy-zoo/galaxy-zoo/data/labels.csv"),
    transform=transform
)

# Get the first image and its label from the dataset
image, label = dataset[0]

# Print the shape of the image tensor and the label
print(image.shape)
