from torch import nn

from src.training.hyperparameters import HyperParamConf


class KLDivLoss(nn.Module):
    """Returns K-L Divergence loss as proposed by Peng et al. 2021 for brain age predicition
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """

    def __init__(self, hp=HyperParamConf(idx=0)):
        super().__init__()
        self.loss_func = nn.KLDivLoss(reduction="none")
        self.use_weight = hp.weighted_loss

    def __call__(self, x, y, hard_label):
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
    """Returns K-L Divergence loss as proposed by Peng et al. 2021 for brain age predicition
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """

    def __init__(self, hp=HyperParamConf(idx=0)):
        super().__init__()
        self.loss_func = nn.MSELoss(reduction="none")
        self.use_weight = hp.weighted_loss

    def __call__(self, x, y):
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
