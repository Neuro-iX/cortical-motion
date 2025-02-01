import click

from src.training.combine_task import launch_training
from src.training.denoise_task import launch_denoise_training
from src.training.regression_task import launch_regression_training
from src.utils.slurm import slurm_adaptor


@click.group()
def train():
    """Command group for data processing"""


@train.command()
@slurm_adaptor(n_cpus=6, mem="490G", n_gpus=4, time="48:00:00")
@click.option(
    "-m",
    "--model",
    type=click.Choice(["SUnet", "ImUnet", "SUnetMap"], case_sensitive=True),
    required=True,
)
def combine(model: str):
    """Launch basic training for Combine and Regression task"""
    launch_training(model)


@train.command()
@slurm_adaptor(n_cpus=10, mem="160G", n_gpus=4, time="24:00:00")
def denoise():
    """Launch basic training for Combine and Regression task"""
    launch_denoise_training()


@train.command()
@slurm_adaptor(n_cpus=10, mem="490G", n_gpus=4, time="48:00:00")
@click.option(
    "-m",
    "--model",
    type=click.Choice(
        ["SFCN", "ImSFCN", "SUnet", "ImUnet", "Opti"], case_sensitive=True
    ),
    required=True,
)
def regression(model: str):
    """Launch basic training for Combine and Regression task"""
    launch_regression_training(model)
