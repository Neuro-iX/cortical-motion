"""
Module to define the Simple Fully Convolutionnal Network from
Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021).
Accurate brain age prediction with lightweight deep neural networks.
Medical Image Analysis, 68, 101871. https://doi.org/10.1016/j.media.2020.101871
"""

from collections.abc import Sequence

import torch
from torch import nn

from src.networks.blocks import MBConv


class MBSFCNBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel, pool=True):
        super().__init__()
        self.block = nn.Sequential(MBConv(kernel_size, in_channel, out_channel))
        if pool:
            self.block.append(nn.MaxPool3d(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        return self.block(x)


class ImprovedSFCNHeadBlock(nn.Sequential):
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


class ImprovedSFCNEncoder(nn.Module):
    """SFCN Encoder for SFCN Model"""

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            MBSFCNBlock(3, 1, 32),
            MBSFCNBlock(3, 32, 64),
            MBSFCNBlock(3, 64, 128),
            MBSFCNBlock(3, 128, 256),
            MBSFCNBlock(3, 256, 256),
            MBSFCNBlock(1, 256, 64, pool=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoding of the SFCN encoder

        Args:
            x (torch.Tensor): input volume tensor

        Returns:
            torch.Tensor: volume's encoding
        """
        return self.convs(x)


class ImprovedSFCNClassifier(nn.Module):
    """SFCN Classifier for the SFCN Model"""

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.classifier = ImprovedSFCNHeadBlock(64, self.num_classes, self.dropout_rate)

    def forward(self, x):
        return self.classifier(x)


class ImprovedSFCNModel(nn.Module):
    """
    Implementation of the model from Han Peng et al. in
    "Accurate brain age prediction with lightweight deep neural networks"
    https://doi.org/10.1016/j.media.2020.101871
    """

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__()
        self.im_shape = im_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.encoder = ImprovedSFCNEncoder()
        self.classifier = ImprovedSFCNClassifier(self.num_classes, self.dropout_rate)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.logsoft(self.classifier(self.encoder(x)))

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = ImprovedSFCNClassifier(self.num_classes, self.dropout_rate)
