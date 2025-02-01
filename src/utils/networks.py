import sys

from lightning import Trainer
from torch import nn


def init_weights(model: nn.Module):
    """Initialize weight for networks using Convolution or Linear layers
      (no transformers) using kaiming / He init

    Args:
        model (nn.Module): Module to apply init strategy
    """
    if isinstance(model, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(model.weight)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm3d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)
        if model.running_mean.isnan().any():
            model.running_mean.fill_(0)
        if model.running_var.isnan().any():
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
