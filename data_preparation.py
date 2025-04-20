# TO DO: prepare the data for each task

from pathlib import Path
from galaxy_classification.data import *
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os


transform = transforms.Compose([
    transforms.ToTensor(), 
])

# Load the dataset
galaxy_dataset = load_image_dataset(Path("data/images/images_training_rev1"), Path("data/labels.csv"))

# Load the dataframe 
labels_df = galaxy_dataset.labels_df

# Now we need to filter the dataset for the different tasks that we need to solve

# TASK 1: classification model to determine if a galaxy is smooth, has a disk or the image is flawed

# Choose the galaxies that have one attribute bigger than 0.8
question1 = labels_df[labels_df[["Class1.1", "Class1.2", "Class1.3"]].max(axis=1) > 0.8][["Class1.1", "Class1.2", "Class1.3"]]
# The max value becomes 1 and the rest 0
question1 = (question1.eq(question1.max(axis=1), axis=0)).astype(int)

# TASK 2: Regression. Could this be a disk? How round is the smooth galaxy?
question2 = labels_df[ ["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]]

# TASK 3: Regression. Is there anything odd about the galaxy? What is the odd feature?
question3 = labels_df[ ["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3", "Class6.1", "Class6.2", "Class8.1", "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6", "Class8.7"]]

# TASK 4: informed regression
question4 = labels_df[labels_df[["Class1.1", "Class1.2", "Class1.3"]].max(axis=1) > 0.8][["Class1.1", "Class1.2", "Class1.3"]].copy()
question4 = (question1.eq(question4.max(axis=1), axis=0)).astype(int)

question4 = pd.concat([question4, labels_df[["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]]], axis=1)

# Save the labels
os.makedirs("data/exercise_1", exist_ok=True)
question1.to_csv("data/exercise_1/labels.csv")
os.makedirs("data/exercise_2", exist_ok=True)
question2.to_csv("data/exercise_2/labels.csv")
os.makedirs("data/exercise_3", exist_ok=True)
question3.to_csv("data/exercise_3/labels.csv")
os.makedirs("data/exercise_4", exist_ok=True)
question4.to_csv("data/exercise_4/labels.csv")
