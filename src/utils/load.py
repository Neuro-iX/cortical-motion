import logging

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Transform,
)


def plot_mri(mri):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(np.rot90(mri[0, 100, :, :]), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(mri[0, :, 100, :]), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(mri[0, :, :, 100]), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


class LogData(Transform):
    def __init__(self, keys="data"):
        self.keys = keys

    def __call__(self, data):
        logging.info(data[self.keys])
        return data


class PlotData(Transform):
    def __init__(self, keys="data"):
        self.keys = keys

    def __call__(self, data):
        matplotlib.use("TkAgg")
        plot_mri(data[self.keys])
        return data


class LoadVolume(Compose):
    """Transform to load data in finetune process"""

    def __init__(self):
        keys = "data"

        tsf = [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
            Orientationd(keys=keys, axcodes="RAS"),
            CenterSpatialCropd(keys=keys, roi_size=(160, 192, 160)),
        ]

        tsf.append(ScaleIntensityd(keys=keys, minv=0, maxv=1))

        super().__init__(tsf)
