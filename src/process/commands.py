import glob
import logging

import click
import nibabel as nib
import pandas as pd
import tqdm

from src import config
from src.process.freesurfer import run_freesurfer_cortical_thichness
from src.process.generate.generate import launch_generate_data
from src.process.motion_estimator import estimate_motion_bids, estimate_motion_test
from src.process.pipeline_inference import add_session
from src.utils.bids import BIDSDirectory, ClinicaDirectory
from src.utils.slurm import slurm_adaptor


@click.group()
def process():
    """Command group for data processing"""


@process.command()
@click.option("-d", "--dataset", type=str, default=None)
def fs(dataset: str):
    """Compute freesurfer cortcial thickness stats"""
    ds = ClinicaDirectory(dataset)
    print(len(list(ds.walk())))
    for sub, ses in list(ds.walk())[:1000]:
        run_freesurfer_cortical_thichness(sub, ses, ds)


@process.command()
@click.option("-O", "--orig", is_flag=True)
@click.option("-P", "--to_process", type=str, default="None")
def synth_fs(orig: bool, to_process: str):
    """Compute freesurfer cortcial thickness stats"""
    fs_synth = BIDSDirectory.FS_SYNTH()

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
@click.option("-n", "--name")
def generate_data(name):
    """Create synthetic motion data from HCPDev"""
    launch_generate_data(
        ClinicaDirectory.CBICCUNY(),
        config.SYNTH_FOLDER if name is None else name,
        config.DATASET_ROOT,
        config.NUM_ITERATIONS,
    )


@process.command()
@slurm_adaptor(n_cpus=64, mem="249G", time="5:00:00")
def generate_freesurfer_data():
    """Create synthetic motion data from HCPDev"""
    launch_generate_data(
        ClinicaDirectory.CBICCUNY(),
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
    """Quantify motion given a dataset name and a model name"""
    estimate_motion_bids(ClinicaDirectory(dataset), model_str=model)


@process.command()
@slurm_adaptor(n_cpus=6, n_gpus=1, mem="60G", time="2:30:00")
@click.option("-m", "--model", type=str, default=None)
def test_model(model: str):
    logging.info("testing model on synth data")
    estimate_motion_test(model)

    dataset_list = [
        "Site-RU_preproc",
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
    ds: BIDSDirectory = BIDSDirectory(dataset)
    for sub, ses in tqdm.tqdm(ds.walk()):
        try:
            t1 = ds.get_T1w(sub, ses)
            nib.load(t1).get_fdata()
        except:
            logging.info(f"missing {sub}, {ses}, {t1}")


@process.command()
def add_ses_openneuro():
    for dataset in glob.glob(
        "OpenNeuro/ds*",
        root_dir=config.DATASET_ROOT,
    ):
        logging.info(f"Starting processing OpenNeuro dataset {dataset}")
        openneuro_ds_dir = BIDSDirectory(dataset)
        add_session(openneuro_ds_dir)


@process.command()
@click.option("-d", "--dataset", type=str, help="dataset name in DATASET_ROOT")
def add_ses(dataset):
    logging.info(f"Starting processing dataset {dataset}")
    openneuro_ds_dir = BIDSDirectory(dataset)
    add_session(openneuro_ds_dir)


@process.command()
def openneuro_fs():
    """Compute freesurfer cortcial thickness stats"""
    for dataset in glob.glob(
        "OpenNeuro_preproc/ds*",
        root_dir=config.DATASET_ROOT,
    )[10:]:
        openneuro_ds_dir = ClinicaDirectory(dataset)
        for sub, ses in list(openneuro_ds_dir.walk()):
            run_freesurfer_cortical_thichness(sub, ses, openneuro_ds_dir)


@process.command()
@click.option("-m", "--model", type=str, default=None)
def openneuro_quant_motion(model):
    for dataset in glob.glob(
        "OpenNeuro_preproc/ds*",
        root_dir=config.DATASET_ROOT,
    ):
        openneuro_ds_dir = ClinicaDirectory(dataset)
        estimate_motion_bids(openneuro_ds_dir, model)
