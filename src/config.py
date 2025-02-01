import os

import dotenv

# Load env variables
dotenv.load_dotenv(override=True)
COMET_API_KEY = os.getenv("COMET_API_KEY")

# Resources
NUM_PROCS = int(os.getenv("NUM_PROCS", "64"))

# Volume Parameters
VOLUME_SHAPE = (160, 192, 160)

# Network Parameters
DROPOUT = 0.6

# Generation Parameters
INTENSITY_SCALING = (0.9, 1.1)
NUM_ITERATIONS = 100
FREESURFER_NUM_ITERATIONS = 50
FREESURFER_NUM_SUBJECTS = 40

# Data Structure
LABEL_KEY = "label"
DATA_KEY = "data"
CLEAR_KEY = "clear"
HARD_LABEL_KEY = "hard_label"

# Datasets
DATASET_ROOT = os.getenv("DATASET_ROOT", ".")
HCPDEV_FOLDER = os.getenv("HCPDEV_FOLDER", "HCP-D_bids")
HBNCIBC_FOLDER = os.getenv("HBNCIBC_FOLDER", "Site-CBIC_preproc")
HBNCUNY_FOLDER = os.getenv("HBNCUNY_FOLDER", "Site-CUNY_preproc")
CBICCUNY_FOLDER = os.getenv("CBICCUNY_FOLDER", "CBIC_CUNY_preproc")
MRART_FOLDER = os.getenv("MRART_FOLDER", "MRART-bids")
SYNTH_FOLDER = os.getenv("SYNTH_FOLDER", "SynthCortical")
FREESURFER_SYNTH_FOLDER = os.getenv("FREESURFER_SYNTH_FOLDER", "FS_SynthCortical_V2")

# Narval
DEFAULT_SLURM_ACCOUNT = "ctb-sbouix"
PATH_FREESURFER_NARVAL = os.getenv("PATH_FREESURFER_NARVAL")

## MOTION MM PARAMETERS
MOTION_N_BINS = 50
MOTION_BIN_RANGE = (-0.4, 2.4)
MOTION_BIN_STEP = (MOTION_BIN_RANGE[1] - MOTION_BIN_RANGE[0]) / MOTION_N_BINS


## ISBI Motion Model
MOTION_MODEL_PATH = "src/process/motion_model/motion_estimator.pt"
