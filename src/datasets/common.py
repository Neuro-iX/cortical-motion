import abc
import os
from typing import Callable, Self

import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


class BaseDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Base dataset with class method to setup from env variables
    It assumes that the dataset is defined by a csv file
    """

    dataset_dir: str
    csv_path: str
    csv_data: pd.DataFrame
    volumes: list[str]
    labels: list[str]
    to_keep: list[str]
    renaming_map: dict[str, str]

    def __init__(self, dataset_root: str, transform: Callable | None = None):
        self.dataset_root = dataset_root
        self.transform = transform
        self.path = os.path.join(self.dataset_root, self.dataset_dir)

        if ".tsv" in self.csv_path:
            self.csv_data = pd.read_csv(
                os.path.join(self.path, self.csv_path), sep="\t"
            )
        else:
            self.csv_data = pd.read_csv(os.path.join(self.path, self.csv_path))

        self.rename_fields()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        item = self.csv_data.iloc[idx]

        dict_item = {}
        for vol_key in self.volumes:
            path_to_vol = os.path.join(self.path, item[vol_key])
            dict_item[vol_key] = self.load_volume(vol_key, path_to_vol)

        for lab_key in self.labels:
            dict_item[lab_key] = self.prepare_label(lab_key, item[lab_key])

        for keep_key in self.to_keep:
            dict_item[keep_key] = item[keep_key]

        return dict_item

    def rename_fields(self):
        """Simple method for renaming logic"""
        self.csv_data = self.csv_data.rename(self.renaming_map)

    def load_volume(self, vol_key: str, vol_path: str) -> torch.Tensor:
        """Should load the volume with the needed transforms

        Args:
            vol_key (str): Key for the volumes in csv file
            vol_path (str): Path to the volume

        Returns:
            torch.Tensor: loaded volume
        """
        if self.transform is not None:
            return self.transform(vol_path)
        else:
            return vol_path

    @abc.abstractmethod
    def prepare_label(
        self, lab_key: str, lab_value: str
    ) -> torch.IntTensor | torch.FloatTensor:
        """Should load the volume with the needed transforms

        Args:
            vol_key (str): Key for the volumes in csv file
            vol_path (str): Path to the volume

        Returns: Prepared label
        """

    @classmethod
    def from_env(cls, transform: Callable | None = None) -> Self:
        """Parameter datasest path with env variables

        Args:
            transform (Callable | None, optional): Transform to apply.
              Defaults to None.

        Returns:
            Self: Dataset
        """
        return cls(config.DATASET_ROOT, transform)
