"""
Module to define the Simple Fully Convolutionnal Network from
Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021).
Accurate brain age prediction with lightweight deep neural networks.
Medical Image Analysis, 68, 101871. https://doi.org/10.1016/j.media.2020.101871
"""

from collections.abc import Sequence

import torch
from monai.networks.blocks import UpSample
from torch import nn

from src.networks.blocks import DownBlock
from src.utils.networks import init_weights


class UpBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel):
        super().__init__()
        self.up = UpSample(3, scale_factor=2, mode="nontrainable")

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

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        up = self.up(torch.concat((x, skip), dim=1))

        c1 = self.conv1(up)
        c2 = self.conv2(c1)
        return c2


class Bottleneck(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super().__init__(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )


class SUnetHeadBlock(nn.Sequential):
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


class SUnetClassifier(nn.Module):
    """SFCN Classifier for the SFCN Model"""

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.down = DownBlock(3, 256, 256)
        self.classifier = nn.Sequential(
            Bottleneck(256, 64),
            SUnetHeadBlock(64, self.num_classes, self.dropout_rate),
        )

    def forward(self, x):
        y = self.down(x)
        return self.classifier(y)


class SUnetModel(nn.Module):
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

        self.encoder_level_1 = DownBlock(3, 1, 32)
        self.encoder_level_2 = DownBlock(3, 32, 64)
        self.encoder_level_3 = DownBlock(3, 64, 128)
        self.encoder_level_4 = DownBlock(3, 128, 256)
        self.bottleneck = Bottleneck(256, 256)
        self.decoder_level_4 = UpBlock(3, 256 + 256, 128)
        self.decoder_level_3 = UpBlock(3, 128 + 128, 64)
        self.decoder_level_2 = UpBlock(3, 64 + 64, 32)
        self.decoder_level_1 = UpBlock(3, 32 + 32, 1)
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding="same",
            ),
            nn.Sigmoid(),
        )
        self.classifier = SUnetClassifier(self.num_classes, self.dropout_rate)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        enc_1 = self.encoder_level_1(x)
        enc_2 = self.encoder_level_2(enc_1)
        enc_3 = self.encoder_level_3(enc_2)
        enc_4 = self.encoder_level_4(enc_3)
        btlnck = self.bottleneck(enc_4)
        dec_4 = self.decoder_level_4(btlnck, enc_4)
        dec_3 = self.decoder_level_3(dec_4, enc_3)
        dec_2 = self.decoder_level_2(dec_3, enc_2)
        dec_1 = self.decoder_level_1(dec_2, enc_1)
        out = self.final_conv(dec_1)
        raw_classes = self.classifier(enc_4)
        logsoft_classes = self.logsoft(raw_classes)
        return out, logsoft_classes, enc_4

    @property
    def latent_shape(self):
        if self._latent_shape is None:
            x = torch.empty((1, *self.im_shape))
            x = self.encoder_level_1(x)
            x = self.encoder_level_2(x)
            x = self.encoder_level_3(x)
            x = self.encoder_level_4(x)
            self._latent_shape = x.shape
        return self._latent_shape[1:]

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = SUnetClassifier(self.num_classes, self.dropout_rate)


class SUnetModelMap(nn.Module):
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

        self.encoder_level_1 = DownBlock(3, 1, 32)
        self.encoder_level_2 = DownBlock(3, 32, 64)
        self.encoder_level_3 = DownBlock(3, 64, 128)
        self.encoder_level_4 = DownBlock(3, 128, 256)
        self.bottleneck = Bottleneck(256, 256)
        self.decoder_level_4 = UpBlock(3, 256 + 256, 128)
        self.decoder_level_3 = UpBlock(3, 128 + 128, 64)
        self.decoder_level_2 = UpBlock(3, 64 + 64, 32)
        self.decoder_level_1 = UpBlock(3, 32 + 32, 1)
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding="same",
            ),
        )
        self.tanh = nn.Tanh()
        self.classifier = SUnetClassifier(self.num_classes, self.dropout_rate)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        enc_1 = self.encoder_level_1(x)
        enc_2 = self.encoder_level_2(enc_1)
        enc_3 = self.encoder_level_3(enc_2)
        enc_4 = self.encoder_level_4(enc_3)
        btlnck = self.bottleneck(enc_4)
        dec_4 = self.decoder_level_4(btlnck, enc_4)
        dec_3 = self.decoder_level_3(dec_4, enc_3)
        dec_2 = self.decoder_level_2(dec_3, enc_2)
        dec_1 = self.decoder_level_1(dec_2, enc_1)
        raw_den = self.final_conv(dec_1)
        denoise = self.tanh(raw_den)
        out = denoise + x
        raw_classes = self.classifier(enc_4)
        logsoft_classes = self.logsoft(raw_classes)
        return out, logsoft_classes, denoise

    @property
    def latent_shape(self):
        if self._latent_shape is None:
            x = torch.empty((1, *self.im_shape))
            x = self.encoder_level_1(x)
            x = self.encoder_level_2(x)
            x = self.encoder_level_3(x)
            x = self.encoder_level_4(x)
            self._latent_shape = x.shape
        return self._latent_shape[1:]

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = SUnetClassifier(self.num_classes, self.dropout_rate)


class SUnetModelRegressor(nn.Module):
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

        self.encoder_level_1 = DownBlock(3, 1, 32)
        self.encoder_level_2 = DownBlock(3, 32, 64)
        self.encoder_level_3 = DownBlock(3, 64, 128)
        self.encoder_level_4 = DownBlock(3, 128, 256)

        self.classifier = SUnetClassifier(self.num_classes, self.dropout_rate)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        enc_1 = self.encoder_level_1(x)
        enc_2 = self.encoder_level_2(enc_1)
        enc_3 = self.encoder_level_3(enc_2)
        enc_4 = self.encoder_level_4(enc_3)

        raw_classes = self.classifier(enc_4)
        logsoft_classes = self.logsoft(raw_classes)
        return logsoft_classes

    @property
    def latent_shape(self):
        if self._latent_shape is None:
            x = torch.empty((1, *self.im_shape))
            x = self.encoder_level_1(x)
            x = self.encoder_level_2(x)
            x = self.encoder_level_3(x)
            x = self.encoder_level_4(x)
            self._latent_shape = x.shape
        return self._latent_shape[1:]

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = SUnetClassifier(self.num_classes, self.dropout_rate)
