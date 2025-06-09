# Estimation of Head Motion in Structural MRI and its Impact on Cortical Thickness Measurements in Retrospective Data

Code used for our article submitted in *Human Brain Mapping*, already available on [Arxiv](https://arxiv.org/abs/2505.23916).  
This repository can be used to train new models and replicate our study.

For inference purposes, you may use [*Agitation*](https://github.com/Neuro-iX/agitation): our motion quantification tool, available as a CLI, Docker container, Boutiques descriptor, Nipoppy pipeline, and Python library.

## Abstract

Motion-related artifacts are inevitable in Magnetic Resonance Imaging (MRI)
and can bias automated neuroanatomical metrics such as cortical thickness. Manual review
cannot objectively quantify motion in anatomical scans, and existing automated approaches
often require specialized hardware or noisy manual labels. Here, we train a 3D convolutional
network to regress motion severity using only synthetically corrupted volumes. We validate
our approach on one held-out site from our training cohort and on 14 fully independent datasets, including
one with manual ratings, achieving a representative R² = 0.65 versus manual labels and
significant thickness–motion correlations in 12 out of 15 datasets. Furthermore, our predicted mo-
tion correlates with subject age in line with prior studies. Our approach generalizes across
scanner brands and protocols, enabling objective, scalable motion assessment in structural
MRI studies without prospective motion correction.

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone <URL>
    ```

2. Set up a new Python environment using conda, venv, or any other tool (we used Python 3.11).  
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Setting Up `.env`

In `src/config.py`, you can easily find which variables are loaded from the `.env` file.  
We provide a `.env.example` file.

### Dataset Organization

In this project, we expect all dataset folders to be at the same depth in the same root folder.  
This allows us to specify only the dataset folder name in our commands instead of the full path.  
The root folder can be specified as an environment variable.

### Executing the Program

There are multiple operations defined in our codebase related to processing and training.  
The main file to execute any operation is `cli.py`.

Most of our commands include an `-S` flag that automatically schedules a job on a SLURM cluster.  

#### Processing

In the `process` command sub-group, you will find:

- `add-ses`: Adds a session layer to non-compliant datasets.  
- `fs`: Computes Freesurfer cortical metrics on a dataset. This command relies on our SLURM environment, which might be difficult to reproduce.  
- `generate-data`: Uses our synthetic data pipeline on any dataset.  
- `generate-freesurfer-data`: Similar to `generate-data` but does not apply elastic deformation or noise, allowing full comparison.  
- `quant-motion`: Uses a trained model on a given dataset to quantify motion on every volume.  
- `test-model`: Tests a given model on all available data.  

You will also find versions of these commands adapted for OpenNeuro datasets, avoiding the need to launch the command separately for each dataset.

#### Training

There are two commands to train a model. The main one is `regression`:

```bash
python cli.py train regression
```

This command launches a single training run with specified hyperparameters.  
If no values are provided, it uses those from our best model in the paper.

The second command is `tune`, which allows launching multiple hyperparameter variations around a common theme (`TUNING_TASK`).  
This command also relies on a SLURM cluster.

## Structure

In this repository, you will find:

- `article`: All data and code used for the paper  
    - `figures`: Figures used in the paper  
    - `freesurfer_outputs`: Freesurfer cortical thickness data for all datasets except HCPs  
    - `model_scripted`: Our best model in TorchScript format  
    - `models`: Models used in the paper  
    - `participants_csv`: Available data about subject age/sex and data split for HBN sites used in data generation  
    - `reports`: Motion prediction reports for each dataset/model  
- `notebooks`: A notebook used to compute and display basic statistics on the train/val/test split of the synthetic dataset  
- `scripts`: Simple scripts and SLURM jobs used for preprocessing data or postprocessing Freesurfer metrics  
- `src`: The core of the project  
    - `dataset`: Defines datasets for training; includes a CSV-based, easily extensible dataset class  
    - `networks`: Contains our flexible implementation of SFCN  
    - `process`: Contains code related to data processing  
    - `training`: Defines Lightning tasks and the hyperparameter class  
    - `utils`: A collection of utility functions or classes grouped by theme  
    - `config.py`: Loads environment variables and defines constant variables  

## Help

For each command, use the `--help` argument to see available options, their purposes, and their data types.

## Tools Used for Project Quality

- `ruff`  
- `isort`  
- `ssort`  
- `pylint` (to double-check `ruff` linting)  
- `mypy`  
- `pydocstyle`  

## Authors

Charles Bricout, Sylvain Bouix, Samira Ebrahimi Kahou.

## Acknowledgments

We reused code from Deep-MI to quantify motion magnitude and from Han-Peng for soft labeling:

- [head-motion-tools](https://github.com/Deep-MI/head-motion-tools/tree/main)  
- [soft labeling](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/tree/master)
