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

    dataset_dir: str | list[str]
    csv_path: str | list[str]
    csv_data: pd.DataFrame
    volumes: list[str]
    labels: list[str]
    to_keep: list[str] = []
    to_augment: list[str] = []
    to_normalize: list[str] = []
    renaming_map: list[tuple[str, str]]

    group_key: str = "group"
    group: str | None = None

    cache = False

    def __init__(
        self,
        dataset_root: str,
        transform: Callable | None = None,
        augment: Callable | None = None,
        normalize: Callable | None = None,
    ):
        self.dataset_root = dataset_root
        self.transform = transform
        self.augment = augment
        self.normalize = normalize

        list_csv_df = []

        if not isinstance(self.csv_path, list):
            self.csv_path = [self.csv_path]
        if not isinstance(self.dataset_dir, list):
            self.dataset_dir = [self.dataset_dir]
        if len(self.dataset_dir) != len(self.csv_path):
            raise ValueError("csv_data and dataset_dir should be of same length")

        for ds_path, csv_path in zip(self.dataset_dir, self.csv_path):

            if ".tsv" in csv_path:
                df = pd.read_csv(
                    os.path.join(self.dataset_root, ds_path, csv_path), sep="\t"
                )

            else:
                df = pd.read_csv(os.path.join(self.dataset_root, ds_path, csv_path))
            df["base_path"] = os.path.join(self.dataset_root, ds_path)
            list_csv_df.append(df)

        self.csv_data = pd.concat(list_csv_df)
        if self.group is not None:
            self.csv_data = self.csv_data[self.csv_data[self.group_key] == self.group]

        self.rename_fields()

        self.cached_data = [None for _ in range(self.__len__())]

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if self.cache and self.cached_data[idx] is not None:
            return self.cached_data[idx]
        else:
            item = self.csv_data.iloc[idx]

            dict_item = {}
            for vol_key in self.volumes:
                path_to_vol = os.path.join(item["base_path"], item[vol_key])
                dict_item[vol_key] = self.load_volume(vol_key, path_to_vol)

            for vol_key in self.to_augment:
                dict_item = self.apply_augment(dict_item)

            for vol_key in self.to_normalize:
                dict_item[vol_key] = self.normalize(dict_item[vol_key])

            for lab_key in self.labels:
                dict_item[lab_key] = self.prepare_label(lab_key, item[lab_key])

            for keep_key in self.to_keep:
                dict_item[keep_key] = item[keep_key]
            if self.cache:
                self.cached_data[idx] = dict_item
            return dict_item

    def rename_fields(self):
        """Simple method for renaming logic"""
        for old_key, new_key in self.renaming_map:
            self.csv_data[new_key] = self.csv_data[old_key].copy()

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

    def apply_augment(self, dict_data) -> torch.Tensor:
        if self.augment is not None:
            return self.augment(dict_data)
        else:
            return dict_data

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
