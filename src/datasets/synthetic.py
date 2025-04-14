import logging
from typing import Callable, Self

import torch
from lightning import LightningDataModule
from monai.transforms import LoadImage, ScaleIntensity
from torch.utils.data import DataLoader

from src import config
from src.datasets.common import BaseDataset
from src.training.hyperparameters import ClassifierType, HyperParamConf
from src.utils.augment import AugmentMRI
from src.utils.soft_label import ToSoftLabel


class SyntheticDataset(BaseDataset):
    dataset_dir = config.SYNTH_FOLDER
    csv_path = "scores.csv"
    volumes = [config.DATA_KEY]

    labels = [config.LABEL_KEY, config.HARD_LABEL_KEY]
    to_normalize = [config.DATA_KEY]
    renaming_map = {
        ("data", config.DATA_KEY),
        ("motion_mm", config.LABEL_KEY),
        ("motion_mm", config.HARD_LABEL_KEY),
    }

    def __init__(
        self,
        dataset_root: str,
        transform: Callable | None = None,
        augment: Callable | None = None,
        normalize: Callable | None = None,
        hp: HyperParamConf = None,
    ):
        logging.info(f"Dataset Root : {dataset_root}")
        logging.info(f"Dataset Dir : {config.SYNTH_FOLDER}")

        self.rescale_labels = hp.classifier == ClassifierType.VANILLA_REG
        self.soft_label = ToSoftLabel.hp_config(hp)
        self.dataset_dir = config.SYNTH_FOLDER

        super().__init__(dataset_root, transform, augment, normalize)

    @classmethod
    def from_env(
        cls,
        transform: Callable | None = None,
        augment: Callable | None = None,
        normalize: Callable | None = None,
        hp: HyperParamConf = None,
    ) -> Self:
        """Parameter datasest path with env variables

        Args:
            transform (Callable | None, optional): Transform to apply.
              Defaults to None.

        Returns:
            Self: Dataset
        """
        return cls(config.DATASET_ROOT, transform, augment, normalize, hp)

    def prepare_label(
        self, lab_key: str, lab_value: str
    ) -> torch.IntTensor | torch.FloatTensor:
        if lab_key == config.LABEL_KEY:
            soft_lab = self.soft_label.value_to_softlabel(float(lab_value))
            return soft_lab
        if self.rescale_labels:
            return float(lab_value) / 4  # rescale between 0 and 1
        else:
            return float(lab_value)


class TestSyntheticDataset(SyntheticDataset):
    group = "test"


class ValSyntheticDataset(SyntheticDataset):
    to_augment = [config.DATA_KEY]
    group = "val"


class TrainSyntheticDataset(SyntheticDataset):
    to_augment = [config.DATA_KEY]
    group = "train"


class SyntheticDataModule(LightningDataModule):
    """
    Base lightning data module
    """

    load_tsf = None
    val_ds = None
    train_ds = None

    def __init__(
        self,
        hp: HyperParamConf = None,
        num_workers: int = 9,
        val_batchsize: int = None,
    ):
        super().__init__()
        self.val_gen = torch.Generator().manual_seed(config.SEED)
        self.hp = hp
        self.batch_size = hp.batch_size
        self.val_batchsize = (
            val_batchsize if val_batchsize is not None else hp.batch_size
        )
        self.load_tsf = LoadImage(ensure_channel_first=True, image_only=True)

        self.num_workers = num_workers

    def setup(self, stage: str):
        val_aug = None
        train_aug = None

        if self.hp.augmentation:
            train_aug = AugmentMRI(
                config.DATA_KEY,
                hist_shift=self.hp.hist_shift,
            )

        self.test_ds = TestSyntheticDataset.from_env(
            self.load_tsf,
            normalize=ScaleIntensity(minv=0, maxv=1),
            hp=self.hp,
            augment=val_aug,
        )
        self.val_ds = ValSyntheticDataset.from_env(
            self.load_tsf,
            normalize=ScaleIntensity(minv=0, maxv=1),
            hp=self.hp,
            augment=val_aug,
        )

        self.train_ds = TrainSyntheticDataset.from_env(
            self.load_tsf,
            augment=train_aug,
            normalize=ScaleIntensity(minv=0, maxv=1),
            hp=self.hp,
        )

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
            generator=self.val_gen,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batchsize,
            pin_memory=True,
            num_workers=self.val_batchsize,
            prefetch_factor=2,
        )
