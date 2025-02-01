import torch
from torch import nn


class MBConv(nn.Module):

    def __init__(self, kernel_size, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=4 * in_channel,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm3d(4 * in_channel),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * in_channel,
                out_channels=4 * in_channel,
                kernel_size=kernel_size,
                padding="same",
                groups=4 * in_channel,
            ),
            nn.BatchNorm3d(4 * in_channel),
        )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                in_channels=4 * in_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

        self.squeeze_excite = SqueezeExcite(
            4 * in_channel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c2_excited = self.squeeze_excite(c2)
        return self.conv3(c2_excited)


class SqueezeExcite(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        batch, channel = x.shape[:2]
        excite = self.squeeze_excite(x).view(batch, channel, 1, 1, 1)
        return x * excite.expand_as(x)


class DownBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel, pool=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )
        if pool:
            self.down = nn.MaxPool3d(2, 2)
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        return self.down(c2)


class SFCNHeadBlock(nn.Sequential):
    """SFCN head bloc, used for classification"""

    def __init__(self, in_channel, out_channel, dropout_rate):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=dropout_rate),
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding="same",
            ),
            nn.Flatten(),
        )
