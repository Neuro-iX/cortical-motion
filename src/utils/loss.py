from torch import nn


class KLDivLoss(nn.Module):
    """Returns K-L Divergence loss as proposed by Peng et al. 2021 for brain age predicition
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """

    def __init__(self):
        super().__init__()
        self.loss_func = nn.KLDivLoss(reduction="sum")

    def __call__(self, x, y):
        y += 1e-16
        n = y.shape[0]
        loss = self.loss_func(x, y) / n
        return loss
