"""Module defining transform to use when loading volumes."""

import logging
from typing import Any

import matplotlib
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Transform,
)

from src.utils.plot import plot_mri


class LogData(Transform):
    """Transform to log data."""

    def __init__(self, keys="data"):
        """Initialize data logging.

        Args:
            keys (str, optional): key to log. Defaults to "data".
        """
        self.keys = keys

    def __call__(self, data: Any):
        """Log data.

        Args:
            data (Any): data to log

        Returns:
            Any: input data without modifications
        """
        logging.info(data[self.keys])
        return data


class PlotData(Transform):
    """Transform to plot MRI data."""

    def __init__(self, keys="data"):
        """Initialize data plotting.

        Args:
            keys (str, optional): key to plot. Defaults to "data".
        """
        self.keys = keys

    def __call__(self, data):
        """Plot data.

        Args:
            data (Any): data to log

        Returns:
            Any: input data without modifications
        """
        matplotlib.use("TkAgg")
        plot_mri(data[self.keys])
        return data


class LoadVolume(Compose):
    """Transform to load data in finetune process."""

    def __init__(self):
        """Initialize basic loading."""
        keys = "data"

        tsf = [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
            Orientationd(keys=keys, axcodes="RAS"),
            CenterSpatialCropd(keys=keys, roi_size=(160, 192, 160)),
        ]

        tsf.append(ScaleIntensityd(keys=keys, minv=0, maxv=1))

        super().__init__(tsf)
