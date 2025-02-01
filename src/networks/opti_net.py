from collections.abc import Sequence

import torch
from torch import nn

from src.networks.blocks import DownBlock, MBConv, SFCNHeadBlock, SqueezeExcite


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
        return self.conv3(c2_excited) + x


class OptiBlock(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, n_conv=2, pool=True):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        )
        for i in range(n_conv - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=out_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    nn.BatchNorm3d(out_channel),
                    nn.ReLU(),
                )
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
        c = self.convs(x)
        return self.down(c)


class ImprovedClassifier(nn.Module):
    """SFCN Classifier for the SFCN Model"""

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.down = DownBlock(3, 256, 256)
        # self.classifier = nn.Sequential(
        #     ReflexionBlock(3, 256, 64),
        #     UnetHeadBlock(64, self.num_classes, self.dropout_rate),
        # )

    def forward(self, x):
        y = self.down(x)
        return self.classifier(y)


class OptiNetModel(nn.Module):
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

        self.down_1 = OptiBlock(3, 1, 32)
        self.down_2 = OptiBlock(3, 32, 64)
        self.down_3 = OptiBlock(3, 64, 128, n_conv=3)
        self.down_4 = OptiBlock(3, 128, 256, n_conv=6)
        self.down_5 = OptiBlock(3, 256, 512, n_conv=3)

        self.reflexion = nn.Sequential(
            MBConv(3, 512, 512), MBConv(3, 512, 512), OptiBlock(3, 512, 64, pool=False)
        )

        self.classifier = SFCNHeadBlock(64, num_classes, dropout_rate)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        enc_1 = self.down_1(x)
        enc_2 = self.down_2(enc_1)
        enc_3 = self.down_3(enc_2)
        enc_4 = self.down_4(enc_3)
        enc_5 = self.down_5(enc_4)

        enc = self.reflexion(enc_5)

        raw_classes = self.classifier(enc)
        logsoft_classes = self.logsoft(raw_classes)
        return logsoft_classes

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
