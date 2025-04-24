# Detecting Automatic Anatomical Measurement Bias Linked to Movement Artifact in Structural Brain MRI using Deep Learning

Code used for our Imaging Neuroscience article

## Description

An in-depth paragraph about your project and overview of use.

Link to usable tools

## Getting Started

### Installing

1. Clone the repository:

    ```git clone <URL>```

2. Set up a new Python environment using conda, venv or any other tool (we used 3.11)  
3. Install dependencies:

    ```pip install -r requirements.txt```

### Setup `.env`

In `src/config.py` you can easily find which variables are loaded from the `.env` file. We provide a `.env.example` file.

### Dataset organisation

In this project, we expect all dataset folders to be at the same depth in the same root folder. This allows us to only specify the dataset folder name in our command instead of a full path. The root folder can be specified as an environment variable.

### Executing program

There are multiple operations defined in our codebase related to processing and training.  
The main file to execute any operation is `cli.py`.

Most of our commands include a `-S` flag that automatically schedules a job for the command on a SLURM cluster.  

#### Processing

In the `process` command sub-group you will have access to:
- `add-ses`: add a session layer on non-compliant datasets
- `fs`: compute Freesurfer cortical metrics on a dataset. This command relies on our SLURM environment, which might not be straightforward to reproduce.
- `generate-data`: uses our synthetic data pipeline on any dataset
- `generate-freesurfer-data`: similar to `generate-data` but does not apply any elastic deformation or noise to allow full comparison.
- `quant-motion`: uses a trained model on a given dataset to quantify motion on every volume
- `test-model`: tests a given model on all available data

You will also find declinations of these commands for OpenNeuro datasets that are here to avoid launching the command once per dataset.

#### Training

There are 2 commands to train a model. The main one is `regression`:

```
python cli.py train regression
```
This command launches a single training run given any hyperparameters. For each hyperparameter, if no value is given, it will use the ones used on our best model in the paper.

The second command is `tune`. It allows launching multiple variations of hyperparameters around a common theme (`TUNING_TASK`). This command also relies on a SLURM cluster.

## Structure

In this repository, you will find:

- article: all data and code used for the paper
- notebooks: a single notebook used to compute and display basic stats on the train/val/test split for the synthetic dataset
- scripts: some simple scripts and SLURM jobs used for preprocessing of data or postprocessing of Freesurfer metrics
- src: the heart of the project  
    - dataset: defines datasets for the training process; it contains a CSV-based, easy-to-extend common dataset class.  
    - networks: contains our flexible implementation of SFCN.  
    - process: contains all code related to processing data  
    - training: defines Lightning tasks and hyperparameter class  
    - utils: collection of small functions or classes grouped by themes  
    - `config.py`: the file loading environment variables and defining independent constant variables

## Help

For each command, there is a helper argument (`--help`) explaining what arguments are available, their purposes, and their types.

## Tools used for project quality
- ruff  
- isort  
- ssort  
- pylint (double checks ruff lint)  
- mypy  
- pydocstyle  

## Authors

Charles Bricout

## Acknowledgments

We reused the code from Deep-MI to quantify motion magnitude and from Han-Peng for soft labelling:
* [head-motion-tools](https://github.com/Deep-MI/head-motion-tools/tree/main)
* [soft labelling](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/tree/master)
