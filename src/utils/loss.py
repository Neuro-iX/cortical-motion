"""Module defining custom losses."""

import torch
from torch import nn

from src.training.hyperparameters import HyperParamConf


class KLDivLoss(nn.Module):
    """Returns K-L Divergence loss as proposed by Peng et al. 2021 for brain age predicition.

    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    c) include potential weighing strategy
    """

    def __init__(self, hp=HyperParamConf(idx=0)):
        """Initialize KLDiv loss.

        Args:
            hp (HyperParamConf, optional): Experiment configuration. Defaults to HyperParamConf(idx=0).
        """
        super().__init__()
        self.loss_func = nn.KLDivLoss(reduction="none")
        self.use_weight = hp.weighted_loss

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, hard_label: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence for given prediction.

        Args:
            x (torch.Tensor): predictions (as distribution)
            y (torch.Tensor): labels (as distribution)
            hard_label (torch.Tensor): labels (as float)

        Returns:
            torch.Tensor: loss value
        """
        y += 1e-16
        n = y.shape[0]
        loss = self.loss_func(x, y)
        loss = loss.sum(dim=1)
        if self.use_weight:
            weights = 1 / (hard_label.float() + 0.25)
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() / n
        return loss


class L2Loss(nn.Module):
    """Returns L2 loss.

    Different from the default PyTorch nn.MSELoss by allowing
    a weighing strategy
    """

    def __init__(self, hp=HyperParamConf(idx=0)):
        """Initialize L2Loss.

        Args:
            hp (HyperParamConf, optional): Experiment configuration. Defaults to HyperParamConf(idx=0).
        """
        super().__init__()
        self.loss_func = nn.MSELoss(reduction="none")
        self.use_weight = hp.weighted_loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute L2 for given prediction.

        Args:
            x (torch.Tensor): predictions (float)
            y (torch.Tensor): labels (float)

        Returns:
            torch.Tensor: loss value
        """
        y = y.float()
        loss = self.loss_func(x, y)
        n = y.shape[0]

        if self.use_weight:
            weights = 1 / (y + 0.25)
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() / n
        return loss
