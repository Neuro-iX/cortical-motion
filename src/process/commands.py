import gzip
import logging

import click

from src import config
from src.process.freesurfer import run_freesurfer_cortical_thichness
from src.process.generate.generate import launch_generate_data
from src.process.measurements import aggregate_measurements_bids
from src.process.motion_estimator import estimate_motion_bids, estimate_motion_records
from src.utils.bids import BIDSDirectory
from src.utils.slurm import slurm_adaptor


@click.group()
def process():
    """Command group for data processing"""


@process.command()
def hcpd_fs():
    """Compute freesurfer cortcial thickness stats"""
    hcpdev = BIDSDirectory.HCPDev()
    for sub, ses in hcpdev.walk():
        run_freesurfer_cortical_thichness(sub, ses, hcpdev)


@process.command()
def hcpd_measure():
    """Aggregate measurements in aparc_stats.{rh, lh}.txt"""
    hcpdev = BIDSDirectory.HCPDev()
    aggregate_measurements_bids(hcpdev)


@process.command()
@slurm_adaptor(n_cpus=64, mem="249G", time="4:00:00")
def generate_data():
    """Create synthetic motion data from HCPDev"""
    launch_generate_data(
        BIDSDirectory.HCPDev(),
        config.SYNTH_FOLDER,
        config.DATASET_ROOT,
        config.NUM_ITERATIONS,
    )


@process.command()
@slurm_adaptor(n_cpus=1, n_gpus=1, mem="30G", time="1:00:00")
@click.option("--file", type=str, default=None)
def quant_motion(file: str):
    """Quantify motion for HCPDev"""
    if file is not None:
        estimate_motion_records(file)
    else:
        estimate_motion_bids(BIDSDirectory.MRART())
    # estimate_motion_bids(BIDSDirectory.HBNCBIC())


@process.command()
def check_bids():
    ds: BIDSDirectory = BIDSDirectory.HBNCBIC()
    for sub, ses in ds.walk():
        try:
            t1 = ds.get_T1w(sub, ses)
            gzip.open(t1)
        except:
            logging.info(f"missing {sub}, {ses}")
