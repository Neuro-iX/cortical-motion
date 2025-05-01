"""Module defining commands for training."""

import click

from src.training.hyperparameters import (
    ActivationType,
    ClassifierType,
    DownsampleType,
    HyperParamConf,
    NormType,
    RegressionLossType,
    TuningTask,
)
from src.training.regression_task import launch_regression_training, tune_model
from src.utils.click_commands import ClickEnumType, TupleParamType
from src.utils.slurm import slurm_adaptor


@click.group()
def train():
    """Command group for training models."""


@train.command()
@click.option(
    "-t",
    "--task",
    type=ClickEnumType(TuningTask),
    required=True,
)
def tune(task: TuningTask):
    """Launch basic training for Combine and Regression task."""
    tune_model(task)


@train.command()
@slurm_adaptor(n_cpus=5, mem="490G", n_gpus=4, time="48:00:00", cpy_synth_ds=True)
@click.option("--idx", type=int, default=0)
@click.option("--task", type=ClickEnumType(TuningTask), default=0)
@click.option("--batch-size", type=int)
@click.option("--kernel-size", type=int)
@click.option("--channels", type=TupleParamType())
@click.option("--n-convs", type=TupleParamType())
@click.option("--norm", type=ClickEnumType(NormType))
@click.option("--down", type=ClickEnumType(DownsampleType))
@click.option("--act", type=ClickEnumType(ActivationType))
@click.option("--n-bins", type=int)
@click.option("--kl-beta", type=float)
@click.option("--loss", type=ClickEnumType(RegressionLossType))
@click.option("--classifier", type=ClickEnumType(ClassifierType))
@click.option("--weight-decay", type=float)
@click.option("--beta1", type=float)
@click.option("--beta2", type=float)
@click.option("--epsilon", type=float)
@click.option("--lr", type=float)
@click.option("--fixed-lr", type=bool)
@click.option("--dropout", type=float)
@click.option("--hist-shift", type=bool)
@click.option("--augmentation", type=bool)
@click.option("--soft-label-func", type=str)
@click.option("--weighted-loss", type=bool)
def regression(**kwargs):
    """Launch a training of a regression model using default or given parameters."""
    provided_params = {
        param: values
        for param, values in kwargs.items()
        if values is not None  # Click sets `None` if no value was provided
    }
    conf = HyperParamConf(**provided_params)
    launch_regression_training(conf)
