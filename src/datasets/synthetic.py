import logging

import torch
from lightning import LightningDataModule
from monai.transforms import (
    CenterSpatialCrop,
    Compose,
    LoadImage,
    Orientation,
    ScaleIntensity,
)
from torch.utils.data import DataLoader

from src import config
from src.datasets.common import BaseDataset
from src.utils.soft_label import ToSoftLabel


class SyntheticDataset(BaseDataset):
    dataset_dir = config.SYNTH_FOLDER
    csv_path = "scores.csv"
    volumes = [config.DATA_KEY, config.CLEAR_KEY]
    labels = [config.LABEL_KEY, config.HARD_LABEL_KEY]
    renaming_map = {
        ("data", config.DATA_KEY),
        ("clear", config.CLEAR_KEY),
        ("motion_mm", config.LABEL_KEY),
        ("motion_mm", config.HARD_LABEL_KEY),
    }
    soft_label = ToSoftLabel.motion_config()

    def prepare_label(
        self, lab_key: str, lab_value: str
    ) -> torch.IntTensor | torch.FloatTensor:
        if lab_key == config.LABEL_KEY:
            soft_lab = self.soft_label.value_to_softlabel(float(lab_value))
            return soft_lab
        else:
            return float(lab_value)


class TestSyntheticDataset(SyntheticDataset):
    group = "test"


class ValSyntheticDataset(SyntheticDataset):
    group = "val"


class TrainSyntheticDataset(SyntheticDataset):
    group = "train"


class SyntheticDataModule(LightningDataModule):
    """
    Base lightning data module
    """

    load_tsf = None
    val_ds = None
    train_ds = None

    def __init__(
        self, batch_size: int = 32, num_workers: int = 9, val_batchsize: int = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batchsize = val_batchsize if val_batchsize is not None else batch_size
        self.load_tsf = Compose(
            [
                LoadImage(ensure_channel_first=True, image_only=True),
                Orientation(axcodes="RAS"),
                CenterSpatialCrop(config.VOLUME_SHAPE),
                ScaleIntensity(minv=0, maxv=1),
            ]
        )
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.val_ds = ValSyntheticDataset.from_env(self.load_tsf)
        self.train_ds = TrainSyntheticDataset.from_env(self.load_tsf)
        logging.info(
            "Train dataset contains %d datas  \nVal dataset contains %d",
            len(self.train_ds),
            len(self.val_ds),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batchsize,
            pin_memory=True,
            num_workers=self.val_batchsize,
            prefetch_factor=2,
        )
