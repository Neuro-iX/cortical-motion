"""Module defining commands related to processing."""

import glob
import logging

import click
import nibabel as nib
import pandas as pd
import tqdm
from nibabel.filebasedimages import ImageFileError

from src import config
from src.process.folder_modifier import add_session
from src.process.freesurfer import run_freesurfer_cortical_thichness
from src.process.generate.generate import launch_generate_data
from src.process.motion_estimator import (estimate_motion_bids,
                                          estimate_motion_test)
from src.utils.bids import BIDSDirectory, ClinicaDirectory
from src.utils.slurm import slurm_adaptor


@click.group()
def process():
    """Command group for data processing."""


@process.command()
@click.option("-d", "--dataset", type=str, default=None)
def fs(dataset: str):
    """Compute freesurfer cortical thickness stats."""
    ds = ClinicaDirectory(dataset)
    print(len(list(ds.walk())))
    for sub, ses in list(ds.walk())[:1000]:
        run_freesurfer_cortical_thichness(sub, ses, ds)


@process.command()
@click.option("-O", "--orig", is_flag=True)
@click.option("-P", "--to_process", type=str, default="None")
def synth_fs(orig: bool, to_process: str):
    """Compute freesurfer cortical thickness stats on synthetic data."""
    fs_synth = BIDSDirectory.fs_synth()

    if to_process is not None:
        files = pd.read_csv(to_process, index_col=0)
        for sub, ses, gen in files.to_records(index=False):
            run_freesurfer_cortical_thichness(sub, ses, fs_synth, gen_id=gen)
    else:
        for sub, ses, gen in fs_synth.walk():
            if (gen == "gen-orig" and orig) or (gen != "gen-orig" and not orig):
                run_freesurfer_cortical_thichness(sub, ses, fs_synth, gen_id=gen)


@process.command()
@slurm_adaptor(n_cpus=64, mem="249G", time="30:00:00")
@click.option(
    "-n", "--name", help="Name of the synthetic data folder. Default to config"
)
def generate_data(name):
    """Create synthetic motion data from HBN's CBIC + CUNY sites.

    Volumes are previously filtered
    """
    launch_generate_data(
        ClinicaDirectory.cbic_cuny(),
        config.SYNTH_FOLDER if name is None else name,
        config.DATASET_ROOT,
        config.NUM_ITERATIONS,
    )


@process.command()
@slurm_adaptor(n_cpus=64, mem="249G", time="5:00:00")
def generate_freesurfer_data():
    """Create synthetic motion data from HBN's CBIC + CUNY sites.

    Volumes are previously filtered.
    This data is specific for FreeSurfer analysis, it includes a specific
    configuration and no elastic deformation
    """
    launch_generate_data(
        ClinicaDirectory.cbic_cuny(),
        config.FREESURFER_SYNTH_FOLDER,
        config.DATASET_ROOT,
        config.FREESURFER_NUM_ITERATIONS,
        True,
    )


@process.command()
@slurm_adaptor(n_cpus=16, n_gpus=1, mem="490G", time="1:00:00")
@click.option("-d", "--dataset", type=str)
@click.option("-m", "--model", type=str)
def quant_motion(dataset: str, model: str):
    """Quantify motion given a dataset name and a model name."""
    estimate_motion_bids(ClinicaDirectory(dataset), model_str=model)


@process.command()
@slurm_adaptor(n_cpus=6, n_gpus=1, mem="60G", time="2:30:00")
@click.option("-m", "--model", type=str, default=None)
def test_model(model: str):
    """Test a given model with all datasets.

    Uses :
    - Synthetic Test
    - HBN-RUBIC
    - MR-ART
    - HCPEP
    - HCP-YA
    - All OpenNeuro
    """
    logging.info("testing model on synth data")
    estimate_motion_test(model)

    dataset_list = [
        "HBN_RUBIC_preproc",
        "MRART_preproc",
        "HCPEP_preproc",
        "HCP-YA_preproc",
    ] + glob.glob(
        "OpenNeuro_preproc/ds*",
        root_dir=config.DATASET_ROOT,
    )
    for dataset in tqdm.tqdm(dataset_list):
        estimate_motion_bids(ClinicaDirectory(dataset), model)


@process.command()
@click.option("-d", "--dataset", type=str, help="dataset name in DATASET_ROOT")
def check_bids(dataset: str):
    """Check if BIDS dataset contains all T1 volumes."""
    ds: BIDSDirectory = BIDSDirectory(dataset)
    for sub, ses in tqdm.tqdm(list(ds.walk())):
        try:
            t1 = ds.get_t1w(sub, ses)
            nib.load(t1).get_fdata()
        except (FileNotFoundError, ImageFileError) as e:
            logging.info("missing %s, %s, %s : %s", sub, ses, t1, e)


@process.command()
def add_ses_openneuro():
    """Modify OpenNeuros datasets structures to include session folders."""
    for dataset in glob.glob(
        "OpenNeuro/ds*",
        root_dir=config.DATASET_ROOT,
    ):
        logging.info("Starting processing OpenNeuro dataset %s", dataset)
        openneuro_ds_dir = BIDSDirectory(dataset)
        add_session(openneuro_ds_dir)


@process.command()
@click.option("-d", "--dataset", type=str, help="dataset name in DATASET_ROOT")
def add_ses(dataset):
    """Modify dataset structures to include default session folders."""
    logging.info("Starting processing dataset %s", dataset)
    openneuro_ds_dir = BIDSDirectory(dataset)
    add_session(openneuro_ds_dir)


@process.command()
def openneuro_fs():
    """Compute freesurfer cortical thickness stats on OpenNeuro."""
    for dataset in glob.glob(
        "OpenNeuro_preproc/ds*",
        root_dir=config.DATASET_ROOT,
    )[10:]:
        openneuro_ds_dir = ClinicaDirectory(dataset)
        for sub, ses in list(openneuro_ds_dir.walk()):
            run_freesurfer_cortical_thichness(sub, ses, openneuro_ds_dir)
