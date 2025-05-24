# Galaxy zoo

This project provides a robust framework for galaxy morphological classification using deep learning techniques. Built with PyTorch, it offers a complete pipeline from data loading to model evaluation, with particular attention to reproducibility and flexibility.

## Tasks
### Task 1: Galaxy Type Classification
Objective: Classify galaxies as:
1. Smooth (Class1.1)
2. Disk (Class1.2)
3. Artifact/Flawed (Class1.3)
### Task 2: Disk Characteristics Regression
Objective: Predict disk properties:
- Could this be a disk? (Class2.1, Class2.2)
- Roundness of smooth galaxies (Class7.1, Class7.2, Class7.3)
### Task 3: Anomaly Detection
Objective: Identify odd features:
- General oddness (Class6.1, Class6.2)
- Specific anomaly types (Class8.1 through Class8.7)

## Initial Setup
1. Clone the repository:
```bash
git clone https://github.com/mariard02/galaxy-zoo
cd galaxy-zoo
```
2. Set up the environment:
```bash
python -m venv galaxy-env
source galaxy-env/bin/activate  # Linux/Mac
galaxy-env\Scripts\activate    # Windows
```
3. Install dependencies:
```bash
pip install -e .
```
5. Ensure raw data is placed in 
- `data/images/images_training_rev1/` (images)
- `data/labels.csv` (original labels)
6. Run the preparation script:
```bash 
python data_preparation.py
```
The script organizes outputs into task-specific directories:
```
data/
├── exercise_1/  # Classification task
│   └── labels.csv
├── exercise_2/  # Disk regression
│   └── labels.csv  
├── exercise_3/  # Anomaly detection
   └── labels.csv
```

## Workflow
