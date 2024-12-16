import os

import dotenv

# Load env variables
dotenv.load_dotenv(override=True)

# Volume Parameters
VOLUME_SHAPE = (160, 192, 160)

# Generation Parameters
INTENSITY_SCALING = (0.9, 1.1)
NUM_ITERATIONS = 100

# Datasets
DATASET_ROOT = os.getenv("DATASET_ROOT", ".")
HCPDEV_FOLDER = os.getenv("HCPDEV_FOLDER", "HCP-D_bids")
SYNTH_FOLDER = os.getenv("SYNTH_FOLDER", "SynthCortical")

# Narval
DEFAULT_SLURM_ACCOUNT = "ctb-sbouix"
PATH_FREESURFER_NARVAL = os.getenv("PATH_FREESURFER_NARVAL")

## MOTION MM PARAMETERS
MOTION_N_BINS = 50
MOTION_BIN_RANGE = (-0.8, 4.8)
MOTION_BIN_STEP = (MOTION_BIN_RANGE[1] - MOTION_BIN_RANGE[0]) / MOTION_N_BINS


## ISBI Motion Model
MOTION_MODEL_PATH = "src/process/motion_model/motion_estimator.pt"
