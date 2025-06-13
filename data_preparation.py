from pathlib import Path
from galaxy_classification.data import *
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
import yaml

transform = transforms.Compose([
    transforms.ToTensor(), 
])

# Load the dataset
galaxy_dataset = load_image_dataset(Path("data/images/images_training_rev1"), Path("data/labels.csv"))
labels_df = galaxy_dataset.labels_df

if 'GalaxyID' not in labels_df.columns and labels_df.index.name == 'GalaxyID':
    labels_df = labels_df.reset_index()

# -------------------------------
# TASK 1: Classification model
# -------------------------------
question1 = labels_df[labels_df[["Class1.1", "Class1.2"]].max(axis=1) > 0.8][["GalaxyID", "Class1.1", "Class1.2"]]
question1.loc[:, ["Class1.1", "Class1.2"]] = (question1[["Class1.1", "Class1.2"]].eq(question1[["Class1.1", "Class1.2"]].max(axis=1), axis=0)).astype(int)

# Convert to integers (0 and 1) using map instead of applymap
question1.loc[:, ["Class1.1", "Class1.2"]] = question1[["Class1.1", "Class1.2"]].apply(lambda x: x.map(int))

q1_train, q1_test = train_test_split(question1, test_size=0.1, shuffle=True)

os.makedirs("data/exercise_1", exist_ok=True)
q1_train.to_csv("data/exercise_1/train.csv", index=False)
q1_test.to_csv("data/exercise_1/test.csv", index=False)

# -------------------------------
# TASK 2: Regression (roundness)
# -------------------------------
question2 = labels_df[["GalaxyID", "Class1.1", "Class1.2", "Class1.3", "Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]]
q2_train, q2_test = train_test_split(question2, test_size=0.1, shuffle=True)

os.makedirs("data/exercise_2", exist_ok=True)
q2_train.to_csv("data/exercise_2/train.csv", index=False)
q2_test.to_csv("data/exercise_2/test.csv", index=False)

hierarchy_config_2 = {
    "hierarchy": {
        "class1": {
            "parent": None,
            "num_classes": 3,
        },
        "class2": {
            "parent": "class1.2",
            "num_classes": 2,
        },
        "class7": {
            "parent": "class1.1",
            "num_classes": 3,
        },
    }
}

hierarchy_path_2 = "data/exercise_2/hierarchy.yaml"
with open(hierarchy_path_2, 'w') as f:
        yaml.dump(hierarchy_config_2, f, sort_keys=False, default_flow_style=False)

# -------------------------------
# TASK 3: Regression (odd features)
# -------------------------------
question3 = labels_df[["GalaxyID", "Class1.1", "Class1.2", "Class1.3", "Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3",
                       "Class6.1", "Class6.2", "Class8.1", "Class8.2", "Class8.3", 
                       "Class8.4", "Class8.5", "Class8.6", "Class8.7"]][:10000]
q3_train, q3_test = train_test_split(question3, test_size=0.1, shuffle=True)

os.makedirs("data/exercise_3", exist_ok=True)
q3_train.to_csv("data/exercise_3/train.csv", index=False)
q3_test.to_csv("data/exercise_3/test.csv", index=False)

hierarchy_config_3 = {
        "hierarchy": {
            "class1": {
                "parent": None,
                "num_classes": 3,
                "comment": "Class1.1, Class1.2, Class1.3"
            },
            "class2": {
                "parent": "class1.2",
                "num_classes": 2,
                "comment": "Class2.1, Class2.2"
            },
            "class7": {
                "parent": "class1.1",
                "num_classes": 3,
                "comment": "Class7.1, Class7.2, Class7.3"
            },
            "class6": {
                "parent": ["class1.1", "class1.2"],
                "num_classes": 2,
                "comment": "Class6.1, Class6.2"
            },
            "class8": {
                "parent": "class6.1",
                "num_classes": 7,
                "comment": "Class8.1 a Class8.7"
            }
        }
    }

hierarchy_path_3 = "data/exercise_3/hierarchy.yaml"
with open(hierarchy_path_3, 'w') as f:
        yaml.dump(hierarchy_config_3, f, sort_keys=False, default_flow_style=False)
