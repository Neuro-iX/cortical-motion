import os

import dotenv

# Load env variables
dotenv.load_dotenv(override=True)
COMET_API_KEY = os.getenv("COMET_API_KEY")

# Resources
NUM_PROCS = int(os.getenv("NUM_PROCS", "64"))

# Volume Parameters
VOLUME_SHAPE = (160, 192, 160)

# Generation Parameters
NUM_ITERATIONS = 300
FREESURFER_NUM_ITERATIONS = 50
FREESURFER_NUM_SUBJECTS = 40

# Data Structure
LABEL_KEY = "label"
DATA_KEY = "data"
CLEAR_KEY = "clear"
HARD_LABEL_KEY = "hard_label"

# Datasets
DATASET_ROOT = os.getenv("DATASET_ROOT", ".")
if "$SLURM_TMPDIR" in DATASET_ROOT:
    DATASET_ROOT = DATASET_ROOT.replace(
        "$SLURM_TMPDIR", os.environ.get("SLURM_TMPDIR", "")
    )
CBICCUNY_FOLDER = os.getenv("CBICCUNY_FOLDER", "HBN_CBIC_CUNY_preproc")
SYNTH_FOLDER = os.getenv("SYNTH_FOLDER", "SynthCortical")
FREESURFER_SYNTH_FOLDER = os.getenv("FREESURFER_SYNTH_FOLDER", "FS_SynthCortical_V2")

# Narval
DEFAULT_SLURM_ACCOUNT = "ctb-sbouix"
PATH_FREESURFER_NARVAL = os.getenv("PATH_FREESURFER_NARVAL")

## MOTION MM PARAMETERS
MOTION_N_BINS = 50
MOTION_BIN_RANGE = (-0.5, 4.5)
# MOTION_BIN_RANGE = (-0.4, 1.4)

MOTION_BIN_STEP = (MOTION_BIN_RANGE[1] - MOTION_BIN_RANGE[0]) / MOTION_N_BINS


REPORT_DIR = os.getenv(
    "REPORT_DIR", "/home/cbricout/projects/ctb-sbouix/cbricout/cortical_reports"
)
RAYTUNE_DIR = os.getenv("RAYTUNE_DIR", "/lustre07/scratch/cbricout/ray_results")

SEED = 2025
