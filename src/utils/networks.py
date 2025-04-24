"""Module for network / training utilities"""

import itertools
import sys
from typing import Any

from lightning import Trainer
from torch import nn
import torch


def init_weights(model: nn.Module):
    """Initialize weight for networks using Convolution or Linear layers
      (no transformers) using kaiming / He init

    Args:
        model (nn.Module): Module to apply init strategy
    """
    if isinstance(model, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(model.weight, nonlinearity="relu")
        if isinstance(model.bias, torch.Tensor):
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, (nn.BatchNorm3d, nn.SyncBatchNorm)):
        nn.init.constant_(model.weight, 1)
        if isinstance(model.bias, torch.Tensor):
            nn.init.constant_(model.bias, 0)

        if (
            isinstance(model.running_mean, torch.Tensor)
            and model.running_mean.isnan().any()
        ):
            model.running_mean.fill_(0)
        if (
            isinstance(model.running_var, torch.Tensor)
            and model.running_var.isnan().any()
        ):
            model.running_var.fill_(1)


class EnsureOneProcess:
    """Context to ensure running on one process"""

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def __enter__(self):
        self.trainer.strategy.barrier()
        if not self.trainer.is_global_zero:
            sys.exit(0)

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass


def get_hyperparams(hyperparam_grid: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Returns a hyperparameter combination based on the job_id.

    Parameters:
        job_id (int): An integer between 0 and n (inclusive) where n+1 is the
                      number of hyperparameter combinations.
        hyperparam_grid (dict): Dictionary where keys are hyperparameter names and
                                values are lists of possible values.

    Returns:
        dict: A dictionary representing one hyperparameter combination.
    """
    keys = list(hyperparam_grid.keys())
    values = [hyperparam_grid[key] for key in keys]

    all_combinations = list(itertools.product(*values))

    return list(
        dict(zip(keys, selected_combination))
        for selected_combination in all_combinations
    )
