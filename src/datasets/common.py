"""Module defining abstract class than can be exxtend to implement a dataset."""

import abc
import os
from typing import Any, Callable, Self

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


class BaseDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Base dataset with class method to setup from env variables.

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

    def __len__(self):
        """Return length of csv data."""
        return len(self.csv_data)

    def __init__(
        self,
        dataset_root: str,
        transform: Callable | None = None,
        augment: Callable | None = None,
        normalize: Callable | None = None,
    ):
        """Initialize dataset.

        Args:
            dataset_root (str): Name of folder containing the dataset
            transform (Callable | None, optional): Tranformation to apply. Defaults to None.
            augment (Callable | None, optional): Augmentation to apply. Defaults to None.
            normalize (Callable | None, optional): Normalization (after augmentation). Defaults to None.

        Raises:
            ValueError: In case the current object contains more csv files than dataset folders
            This class assumes one csv per folder
        """
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

    def rename_fields(self):
        """Rename dictionnary keys."""
        for old_key, new_key in self.renaming_map:
            self.csv_data[new_key] = self.csv_data[old_key].copy()

    def load_volume(self, vol_key: str, vol_path: str) -> torch.Tensor | str:
        """Load the volume with the needed transforms.

        Args:
            vol_key (str): Key for the volumes in csv file
            vol_path (str): Path to the volume

        Returns:
            torch.Tensor: loaded volume
        """
        if self.transform is not None:
            return self.transform(vol_path)
        return vol_path

    def apply_augment(self, dict_data: dict[str, Any]) -> dict[str, Any]:
        """Apply augmentation defined in `augment`.

        Args:
            dict_data (dict[str,Any]): Data as a dictionnary

        Returns:
            dict[str,Any]: Dictionnary data with augmentation applied
        """
        if self.augment is not None:
            return self.augment(dict_data)
        return dict_data

    @abc.abstractmethod
    def prepare_label(
        self, lab_key: str, lab_value: str
    ) -> float | torch.Tensor | np.ndarray:
        """Prepqres the label for training.

        Args:
            lab_key (str): Key for the label in csv file
            lab_value (str): current label value

        Returns: Prepared label
        """

    @classmethod
    def from_env(cls, transform: Callable | None = None) -> Self:
        """Parameterize dataset path with env variables.

        Args:
            transform (Callable | None, optional): Transform to apply.
              Defaults to None.

        Returns:
            Self: Dataset
        """
        return cls(config.DATASET_ROOT, transform)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load, Augment and Normalize a row of the csv file.

        Args:
            idx (int): index of row to use

        Returns:
            dict[str,Any]: Dictionnary containing ready to use data
        """
        if self.cache and self.cached_data[idx] is not None:
            return self.cached_data[idx]

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
