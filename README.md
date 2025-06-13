# Galaxy zoo

This project provides a robust framework for galaxy morphological classification using deep learning techniques. Built with PyTorch, it offers a complete pipeline from data loading to model evaluation, with particular attention to reproducibility and flexibility.

## Description of the data
To train our neural network to classify galaxies, we use images and labels from the Galaxy Zoo project, a citizen science initiative where volunteers help classify galaxies based on their shapes.
The dataset consists of color images (in PNG format) of galaxies. For each image, we focus on the central region of the galaxy, which we crop and use as input to our model.

Each galaxy comes with 11 labels, which are probabilities between 0 and 1. These values represent how likely volunteers were to answer "yes" to a series of questions about the galaxy’s appearance—for example, whether it looks smooth, has spiral arms, or is edge-on. All the data can be downloaded from [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data).

An important aspect of these labels is that they follow a hierarchical structure. The questions that volunteers answer are presented in a flow: some questions only appear based on previous answers. For example, the question *“How rounded is it?”* is only asked if the galaxy was first classified as “smooth.” This means that the probabilities for follow-up answers are conditional on earlier ones. So, in this example, the total probability across the roundness options must add up to the probability that the galaxy was identified as smooth.

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
3. Install the module:
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
│   └── test.csv
│   └── train.csv
├── exercise_2/  # Disk regression
│   └── test.csv  
│   └── train.csv  
│   └── hierarchy.yaml  
└── exercise_3/  # Anomaly detection
    └── test.csv  
    └── train.csv   
    └── hierarchy.yaml  
```
In each of the folders, there is a `csv` file with the relevant labels for each task, as well as the ID of each of the galaxies. Additionally, for exercises 2 and 3 we can find a `hierarchy.yaml` file. In it, there is a dictionary that identifies which is the parent class for each of the questions. This file will be required if we are dealing with a regression task.

## Workflow
### Training the model
To train the model, run 
```bash 
python train.py --run_name name_of_the_run
```
This command copies the default configuration file `config_default.yaml` into the folder `outputs/name_of_the_run/`, renaming it to `config.yaml`.
Next, open `outputs/name_of_the_run/config.yaml` and adjust the parameters as needed for your run. Once you've saved your changes, press Enter in the terminal to start training. Detailed instructions on how to edit the configuration file can be found [here](#setting-up-the-configuration).

After training ends, the folder `outputs/name_of_the_run/` will have the following structure:
```
outputs/name_of_the_run/
├── classifier
│   ├── hyperparameters.yaml
│   └── parameters.pth
├── config.yaml
└── plots
    └── training_summary.pdf
```
- The `classifier` folder contains the network's hyperparameters and trained weights, which can be used for evaluation or inference.
- The `plots` folder includes `training_summary.pdf`, which visualizes the evolution of training and validation loss across epochs. For classification tasks, it also shows the accuracy progression during training.

If you don't need to modify the default configuration, you can skip the manual editing step by running:
```bash
python3 train.py --run_name name_of_the_run --no_config_edit
```
This command copies the default configuration to `outputs/name_of_the_run/` and immediately starts the training process without prompting for edits.

If you need help or want to see the available arguments, run:
```bash
python3 train.py --help
```
This will display a description of all supported options.

### Evaluating the model
Once the model has been trained, you can evaluate its performance on unseen data.
- For classification tasks, this includes computing the accuracy, plotting ROC curves, and generating a confusion matrix.
- For regression tasks, the evaluation reports the mean squared error (MSE) and creates histograms comparing the true and predicted label values.

To run the evaluation, use:
```bash
python3 evaluate.py --run_name name_of_the_run
```
Make sure that the `--run_name` value matches the name used during training. The results will be printed in the terminal, and the corresponding plots will be saved in the `outputs/name_of_the_run/plots` folder.

### Setting up the configuration
Before training or evaluating the model, one essential step remains: defining a configuration file that specifies the parameters for both processes. This file, typically named `config.yaml`, allows you to customize the behavior of the training and evaluation scripts without modifying the source code.

Below is an example of a `config.yaml` file:
```
training:
  epoch_count: 15
  batch_size: 128
  learning_rate: 5.e-4
  validation_fraction: 0.2
  data_dir: "data/exercise_2"

  network:
    channel_count_hidden: 16
    convolution_kernel_size: 5
    mlp_hidden_unit_count: 128
    output_units: 8
    task_type: regression
    

evaluation:
  batch_size: 1024
  task_type: regression
  data_dir: "data/exercise_2"
```
**Training Parameters**
+ `epoch_count`: Number of training epochs.
+ `batch_size`: Batch size used during training.
+ `learning_rate`: Learning rate for the optimizer.
+ `validation_fraction`: Fraction of the dataset to be used for validation.
+ `data_dir`: Relative path to the directory containing the files with the labels and the hierarchy in the questions.

**Network Architecture**

The network block defines the structure of the model:

+ `channel_count_hidden`: Sets the base number of hidden channels in the convolutional blocks.
   + In your model, this determines:
      + First convolution block: input → `channel_count_hidden``
      + Second convolution block: `channel_count_hidden` → 2 × `channel_count_hidden`
   + Controls the capacity of feature extraction.
+ `convolution_kernel_size`: Size of the kernel used in the convolutions.
+ `mlp_hidden_unit_count`: Base number of units in the first hidden layer of the MLP.
The architecture expands and contracts around this:
   + First linear: → `mlp_hidden_unit_count`
   + Second linear: → 2 × `mlp_hidden_unit_count`
   + Third linear: → `output_units`
+ `output_units`: Number of output units. For classification, this should match the number of classes.
+ `task_type`: Type of task; valid values are:
   + `regression` for continuous outputs. The hierarchy of the data must be given.
   + `classification_multiclass`: model predicts class probabilities via logits.

**Evaluation Parameters**

The evaluation block mirrors the structure of the training section but is used exclusively when evaluating the model:

+ `batch_size`: Batch size used for evaluation.
+ `task_type`: Same as in training; must match the trained model.
+ `data_dir`: Path to the data used for evaluation (usually the test set).

By editing the configuration file, you can fine-tune the model’s behavior and architecture, allowing you to easily experiment with different setups. The flexibility of this approach makes it ideal for both prototyping and structured experimentation.

### Further options
Up to this point, we have focused on configuring the model by editing the `config.yaml` file, which allows you to train and evaluate different architectures without modifying the source code. This is ideal for quick experimentation and reproducibility. However, more fine-grained control, such as implementing novel architectures, adding attention mechanisms, or integrating new types of output heads, requires editing the model definition directly.

The architecture of the model is implemented in the file `galaxy_classification/networks/cnn.py`.
Here, you can:
- Redesign or extend the CNN feature extractor by modifying the `DoubleConvolutionBlock` or replacing it entirely.
- Adjust or restructure the MLP head, changing the number, size, or type of layers.
- Add new task-specific output layers, such as multitask learning heads, hierarchical regressors, or uncertainty estimates.
- Implement custom forward passes if your task demands multiple inputs, auxiliary outputs, or non-standard loss functions.

In addition to editing the model architecture or configuration file, you can fine-tune the training behavior by modifying a few lines in the train.py script. Specifically, the following block controls the optimizer, loss function, and training loop:
```python 
optimizer = AdamW(network.parameters(), lr=config.learning_rate, weight_decay=5.e-5)

loss = get_loss(config=config, weight=weights, hierarchy_config=hierarchy_config)

print_divider()
print("Training... \n")

training_summary = galaxy_classification.fit(
    network,
    optimizer,
    loss,
    split_dataloader.training_dataloader,
    split_dataloader.validation_dataloader,
    config.epoch_count,
    patience=20,
    delta=1.e-5
)
```
Here’s what you can customize:
- **Weight decay:** The `weight_decay` argument in the optimizer controls L2 regularization, which helps prevent overfitting by penalizing large weights. If you don’t want to use it, you can simply remove the argument or set it to 0.0.
- **Early stopping:** The `patience` and `delta` arguments control early stopping behavior:
   - `patience=20` means training will stop if the validation loss doesn't improve after 20 consecutive epochs.
   - `delta=1.e-5` sets the minimum change in validation loss that qualifies as an improvement.

If you remove patience and delta from the call to fit, early stopping will not be used, and the model will train for the full number of epochs specified.

Before training begins, the input images and labels are loaded and preprocessed. This is handled by the following lines in `train.py`:
```python
galaxy_dataset = load_image_dataset(
    image_dir,
    label_path,
    task=config.network.task_type,
    transform=None  # Optional torchvision transforms can be passed here
)

print("Preprocessing the data. \n")

preprocessor = GalaxyPreprocessor(
    image_dir=image_dir,
    label_path=label_path,
    scale_factor=1.0,
    batch_size=config.batch_size,
    normalize=True,
)
```
Even if `transform=None` is passed when loading the dataset, preprocessing and data augmentation are automatically applied later by the GalaxyPreprocessor class. This includes:
- Cropping and resizing to standardize input size.
- Random augmentations such as:
   - `RandomHorizontalFlip`
   - `RandomVerticalFlip`
   - `RandomRotation`
- Normalization using dataset-wide mean and standard deviation (computed automatically unless disabled).
- Optional scaling of image intensities via `scale_factor`.

All of these steps are bundled into a transform pipeline internally and applied when `calling preprocessor.apply_preprocessing(...)`.

Parameters you can customize:
- `scale_factor`: Adjusts the pixel intensity scale (default is 1.0). 
- `normalize=True`: Enables or disables normalization. Set to `False` to skip the computation and application of mean/std normalization.
- `batch_size`: Affects the batch size used to estimate dataset statistics. Doesn’t influence training directly, but can impact normalization accuracy.

**Note on additional transforms:**
If you need full control over the transform pipeline (e.g. applying grayscale, center crop, or advanced augmentations), you can still bypass the default `GalaxyPreprocessor` by directly providing a custom `transform` to `load_image_dataset`. However, this will override the internal preprocessing.
