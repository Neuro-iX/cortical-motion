"""
Module to define the Simple Fully Convolutionnal Network from
Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021).
Accurate brain age prediction with lightweight deep neural networks.
Medical Image Analysis, 68, 101871. https://doi.org/10.1016/j.media.2020.101871
"""

import torch
from torch import nn

from src.training.hyperparameters import (
    ActivationType,
    ClassifierType,
    DownsampleType,
    HyperParamConf,
    NormType,
)


def get_norm_layer(norm: NormType, channels):
    """Translate enumeration values to norm layer"""
    norm_layer: nn.Module = nn.Identity()
    if norm == NormType.BATCH:
        norm_layer = nn.BatchNorm3d(channels)
    elif norm == NormType.INSTANCE:
        norm_layer = nn.InstanceNorm3d(channels)
    elif norm == NormType.LAYER:
        norm_layer = CNNLayerNorm(channels=channels)
    elif norm == NormType.GROUP:
        norm_layer = nn.GroupNorm(num_groups=1, num_channels=channels)
    return norm_layer


def get_activation(activation: ActivationType):
    """Translate enumeration values to activation layer"""

    if activation == ActivationType.RELU:
        return nn.ReLU()
    if activation == ActivationType.PRELU:
        return nn.PReLU()
    return nn.Identity()


class CNNLayerNorm(nn.Module):
    """Definition of layers norm without shape
    normalizing on channels"""

    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform layer normalization by swapping dimension

        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Normalized input
        """
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class DownBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        n_convs: int,
        norm: NormType,
        down: DownsampleType,
        act: ActivationType,
        kernel_size: int,
    ):
        super().__init__()
        self.convs = nn.Sequential()
        for i in range(n_convs):
            in_ch = out_channel
            if i == 0:
                in_ch = in_channel
            stride = 1
            if down == DownsampleType.STRIDE and i == n_convs - 1:
                stride = 2
            self.convs.append(
                nn.Conv3d(
                    in_channels=in_ch,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    padding="same" if stride == 1 else 1,
                    stride=stride,
                )
            )
            self.convs.append(get_norm_layer(norm, out_channel))
            self.convs.append(get_activation(act))

        if down == DownsampleType.POOL:
            self.convs.append(nn.MaxPool3d(2, 2))
        if down == DownsampleType.AUG_STRIDE:
            self.convs.append(
                nn.Conv3d(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    padding=1,
                    stride=2,
                )
            )
            self.convs.append(get_norm_layer(norm, out_channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        return self.convs(x)


class SFCNHeadBlock(nn.Sequential):
    """SFCN head block"""

    def __init__(self, hp: HyperParamConf):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=hp.dropout),
            nn.Conv3d(
                in_channels=hp.channels[-1],
                out_channels=hp.n_bins,
                kernel_size=1,
                padding="same",
            ),
            nn.Flatten(),
            nn.LogSoftmax(dim=1),
        )


class SFCNLongHeadBlock(nn.Sequential):
    """Longer SFCN head block"""

    def __init__(self, hp: HyperParamConf):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=hp.dropout),
            nn.Conv3d(
                in_channels=hp.channels[-1],
                out_channels=hp.n_bins,
                kernel_size=hp.kernel_size,
                padding="same",
            ),
            get_norm_layer(hp.norm, hp.n_bins),
            get_activation(hp.act),
            nn.Conv3d(
                in_channels=hp.n_bins,
                out_channels=hp.n_bins,
                kernel_size=1,
                padding="same",
            ),
            nn.Flatten(),
            nn.LogSoftmax(dim=1),
        )


class SFCNVanillaRegBlock(nn.Sequential):
    """SFCN regression block"""

    def __init__(self, hp: HyperParamConf):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=hp.dropout),
            nn.Conv3d(
                in_channels=hp.channels[-1],
                out_channels=hp.n_bins,
                kernel_size=hp.kernel_size,
                padding="same",
            ),
            get_norm_layer(hp.norm, hp.n_bins),
            get_activation(hp.act),
            nn.Conv3d(
                in_channels=hp.n_bins,
                out_channels=1,
                kernel_size=1,
                padding="same",
            ),
            nn.Flatten(),
            nn.Sigmoid(),
        )


class SFCNEncoder(nn.Module):
    """SFCN Encoder for SFCN Model"""

    def __init__(self, hp: HyperParamConf):
        super().__init__()
        self.convs = nn.Sequential()
        for i, c in enumerate(hp.n_convs):
            down = hp.down
            if i >= 5:
                down = DownsampleType.NONE
            self.convs.append(
                DownBlock(
                    hp.channels[i],
                    hp.channels[i + 1],
                    c,
                    hp.norm,
                    down,
                    hp.act,
                    hp.kernel_size,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoding of the SFCN encoder

        Args:
            x (torch.Tensor): input volume tensor

        Returns:
            torch.Tensor: volume's encoding
        """
        return self.convs(x)


class GenericSFCNModel(nn.Module):
    """
    Flexible implementation of the model from Han Peng et al. in
    "Accurate brain age prediction with lightweight deep neural networks"
    https://doi.org/10.1016/j.media.2020.101871
    """

    def __init__(
        self,
        hp: HyperParamConf,
    ):
        super().__init__()
        self.hp = hp
        self.encoder = SFCNEncoder(hp)

        if hp.classifier == ClassifierType.SFCN:
            self.classifier: nn.Module = SFCNHeadBlock(hp)
        elif hp.classifier == ClassifierType.SFCN_LONG:
            self.classifier = SFCNLongHeadBlock(hp)
        elif hp.classifier == ClassifierType.VANILLA_REG:
            self.classifier = SFCNVanillaRegBlock(hp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the raw output

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Prediction (bins or float)
        """
        if self.hp.norm == NormType.LAYER:
            x = x.to(memory_format=torch.channels_last_3d)
        return self.classifier(self.encoder(x))
