# Detecting Automatic Anatomical Measurement Bias Linked to Movement Artifact in Structural Brain MRI using Deep Learning

Code used for our Imaging Neuroscience

## Description

An in-depth paragraph about your project and overview of use.

Link to usable tools

## Getting Started

### Installing

1. Clone the repository :

    ```git clone <URL>```

2. Setup a new python environment using conda, venv or any other tool (we used 3.11)
3. Install dependencies :

    ```pip install -r requirements.txt```

### Setup `.env`

In `src/config.py` you can easily find which variables are loaded from the `.env` file, we provide a `.env.example` file.

### Dataset organisation

In this project, we expect all dataset folder to be at the same depth in the same root folder. This allow us to only specify the dataset folder name in our command instead of a full path. The root folder can be specified as an environment variable

### Executing program

There is multiple operation defined in our codebase related to processing and training.
The main file to execute any operation is `cli.py`.

Most of our command include a `-S` flag that automatically schedule a job for the command on a slurm cluster.  

#### Processing

In the `process` command sub-group you will have access to :
- `add-ses`: add a session layer on non compliant datasets
- `fs` : compute freesurfer cortical metrics on a dataset. This command relies on our SLURM environment which might nit be straigth forward to reproduce.
- `generate-data` : which use our synthetic data pipeline on any datasets
- `generate-freesurfer-data` : similar to `generate-data` but does not apply any elastic deformation or noise to allow full comparison.
- `quant-motion` : which use a trained model on a given dataset to quantify motion on every volumes
- `test-model` : Test a given model on all available data

You will also find declination of those command for OpenNeuro datasets that are here to avoid launching the command once per dataset.

#### Training

There is 2 command to train a model. The main one is `regression` :

```
python cli.py train regression
```
This command launch a single training given any hyperparameters. For each hyperparameter, if no value is given, it will use the ones used on our best model in the paper.

The second command is `tune`, it allows to launch multiple variation of hyperparameters around a common theme (`TUNING_TASK`). This command also relies on SLURM cluster.

## Structure

In this repository, you will find :

- article : all data and code used for the paper
- notebooks : a single notebook used to compute and display basic stat on the train/val/test split for the synthetic dataset
- scripts : some simple scripts and slurm jobs used for preprocessing of data or postprocessing of freesurfer metrics
- src : the heart of the project
    - dataset : define dataset for the training process, it contains a csv based, easy to extend common dataset class.
    - networks : contains our flexible implementation of SFCN.
    - process : contains all code related to processing data
    - training : defines lightning tasks and hyperparameter class
    - utils : collection of small function or classes regrouped by themes
    - `config.py` : the file loading environment variable and defining independent constant variables


## Help

For each command there is an helper argument (`--help`) explaining what arguments are availables, their purposes and their types

## Tools used for project quality
- ruff
- isort
- ssort
- pylint (double check ruff lint)
- mypy
- pydocstyle

## Authors

Charles Bricout 

## Acknowledgments

We reused the code from Deep-MI to quantify motion magnitude and from Han-Peng for soft labelling :
* [head-motion-tools](https://github.com/Deep-MI/head-motion-tools/tree/main)
* [soft labelling](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/tree/master)

