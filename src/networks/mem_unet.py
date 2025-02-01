"""
Module to define the Simple Fully Convolutionnal Network from
Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021).
Accurate brain age prediction with lightweight deep neural networks.
Medical Image Analysis, 68, 101871. https://doi.org/10.1016/j.media.2020.101871
"""

import logging
from collections.abc import Sequence

import torch
from monai.networks.blocks import UpSample
from torch import nn


class MBConv(nn.Module):

    def __init__(self, kernel_size, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=4 * channels,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm3d(4 * channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * channels,
                out_channels=4 * channels,
                kernel_size=kernel_size,
                padding="same",
                groups=4 * channels,
            ),
            nn.BatchNorm3d(4 * channels),
        )

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                in_channels=4 * channels,
                out_channels=channels,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
        )

        self.squeeze_excite = SqueezeExcite(
            4 * channels,
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
        return self.conv3(c2_excited) + x


class DownBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel):
        super().__init__()
        self.conv = MBConv(kernel_size=kernel_size, channels=in_channel)
        self.channel_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        c1 = self.conv(x)
        c1 = self.channel_conv(c1)
        return c1


class UpBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel):
        super().__init__()
        self.up = UpSample(
            3,
            scale_factor=2,
            mode="nontrainable",
        )
        self.channel_conv = nn.Conv3d(in_channel, out_channel, 3, padding="same")
        self.conv = MBConv(kernel_size=kernel_size, channels=out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        up = self.up(x)
        c = self.channel_conv(up)
        c = self.conv(c)
        return c


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


class ReflexionBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel, pool=True, upsample=False):
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

        self.squeeze_excite = SqueezeExcite(out_channel)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        c1 = self.conv1(x)
        excited = self.squeeze_excite(c1)
        final = self.conv2(excited)
        return final


class UnetHeadBlock(nn.Sequential):
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


class UnetConvert(nn.Sequential):

    def __init__(self, in_channel, out_channel=None):
        super().__init__(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=in_channel if out_channel is None else out_channel,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm3d(in_channel if out_channel is None else out_channel),
            nn.ReLU(),
        )


class ImprovedClassifier(nn.Module):
    """SFCN Classifier for the SFCN Model"""

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.classifier = nn.Sequential(
            DownBlock(3, 256, 256),
            ReflexionBlock(3, 256, 64),
            UnetHeadBlock(64, self.num_classes, self.dropout_rate),
        )

    def forward(self, x):
        return self.classifier(x)


class MemUnetModel(nn.Module):
    """
    Implementation of the model from Han Peng et al. in
    "Accurate brain age prediction with lightweight deep neural networks"
    https://doi.org/10.1016/j.media.2020.101871
    """

    _latent_shape = None

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__()

        self.im_shape = im_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.down_1 = DownBlock(3, 1, 32)
        self.down_2 = DownBlock(3, 32, 64)
        self.down_3 = nn.Sequential(DownBlock(3, 64, 128), MBConv(3, 128))
        self.down_4 = DownBlock(3, 128, 256)

        self.bottleneck_1 = MBConv(3, 256)

        self.up_4 = nn.Sequential(UpBlock(3, 256 + 256, 128), MBConv(3, 128))
        self.up_3 = UpBlock(3, 128 + 128, 64)
        self.up_2 = UpBlock(3, 64 + 64, 32)
        self.up_1 = UpBlock(3, 32 + 32, 1)

        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=7,
                padding="same",
            ),
            nn.Sigmoid(),
        )

        # self.classifier = ImprovedClassifier(self.num_classes, self.dropout_rate)
        # self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        enc_1 = self.down_1(x)
        enc_2 = self.down_2(enc_1)
        enc_3 = self.down_3(enc_2)
        enc_4 = self.down_4(enc_3)
        btlnck = self.bottleneck_1(enc_4)
        dec_4 = self.up_4(torch.concat((btlnck, enc_4), dim=1))
        dec_3 = self.up_3(torch.concat((dec_4, enc_3), dim=1))
        dec_2 = self.up_2(torch.concat((dec_3, enc_2), dim=1))
        dec_1 = self.up_1(torch.concat((dec_2, enc_1), dim=1))
        final_denoise = self.final_conv(dec_1)
        # raw_classes = self.classifier(enc_4)
        # logsoft_classes = self.logsoft(raw_classes)
        return final_denoise

        # return final_denoise, logsoft_classes, enc_4

    @property
    def latent_shape(self):
        if self._latent_shape is None:
            x = torch.empty((1, *self.im_shape))
            x = self.down_1(x)
            x = self.down_2(x)
            x = self.down_3(x)
            x = self.down_4(x)
            self._latent_shape = x.shape
        return self._latent_shape[1:]

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = ImprovedClassifier(self.num_classes, self.dropout_rate)
