import glob
import logging
import os
import re
from typing import Generator, Self

from src import config


class BIDSDirectory:
    """Utility to query BIDS directories"""

    has_session: bool = False
    has_sites: bool = False
    sites: list[str] = None

    def __init__(self, dataset: str, root_dir: str = config.DATASET_ROOT):
        """Args:
        dataset (str): Dataset directory name
        root_dir (str): Root directory for all datasets
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.base_path = os.path.join(self.root_dir, self.dataset)

    def get_subjects(self) -> list[str]:
        """Retrieve list of all subjects
        Returns:
            list[str]: list of all subjects
        """
        return glob.glob("sub-*", root_dir=self.base_path)

    def get_session(self, sub_id: str) -> list[str]:
        """Retrieve list of seesion
        Args:
            sub_id (str): Subject id (sub-???)
        Returns:
            list[str]: list of session
        """

        return glob.glob("ses-*", root_dir=os.path.join(self.base_path, sub_id))

    def walk(self) -> Generator[tuple[str, str], None, None]:
        """Iterate over every subjects and session of the dataset
        Yields:
            Generator[tuple[str, str], None, None]: sub_id, ses_id
        """
        for sub in self.get_subjects():
            for ses in self.get_session(sub):
                yield sub, ses

    def __len__(self):
        return len(list(self.walk()))

    def get_T1w(self, sub_id: str, ses_id: str = "ses-1") -> str:
        """Return path to T1w volume

        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str, optional): Session id (ses-???). Defaults to "ses-1".

        Returns:
            str: Path to T1w
        """
        # logging.info(os.path.join(self.base_path, sub_id, ses_id, "anat", "*T1w.nii.gz"))
        return glob.glob(
            os.path.join(self.base_path, sub_id, ses_id, "anat", "*T1w.nii.gz")
        )[0]

    def get_all_T1w(self, sub_id: str, ses_id: str = "ses-1") -> str:
        """Return path to T1w volume

        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str, optional): Session id (ses-???). Defaults to "ses-1".

        Returns:
            str: Path to T1w
        """
        # logging.info(os.path.join(self.base_path, sub_id, ses_id, "anat", "*T1w.nii.gz"))
        return glob.glob(
            os.path.join(self.base_path, sub_id, ses_id, "anat", "*T1w.nii.gz")
        )

    def extract_sub_ses(self, path: str) -> tuple[str | None, str | None]:
        """Extract subject and session identifier from path

        Args:
            path (str): path to NifTi file

        Returns:
            tuple[str | None, str | None]: sub-id, ses-id
        """
        req = r".*(sub-[\dA-Za-z]*).*(ses-[\dA-Za-z]*).*"
        match = re.match(req, path)
        sub = None
        ses = None
        if match is not None:
            groups = match.groups()
            if len(groups) == 2:
                sub, ses = groups
            else:
                sub = groups[0]
        return sub, ses

    @staticmethod
    def HCPDev() -> Self:
        """Method to create object for HCP Dev"""
        return BIDSDirectory(config.HCPDEV_FOLDER, root_dir=config.DATASET_ROOT)

    @staticmethod
    def HBNCBIC() -> Self:
        """Method to create object for HBN CIBC site"""
        return BIDSDirectory(config.HBNCIBC_FOLDER, root_dir=config.DATASET_ROOT)

    @staticmethod
    def HBNCUNY() -> Self:
        """Method to create object for HBN CUNY site"""
        return BIDSDirectory(config.HBNCUNY_FOLDER, root_dir=config.DATASET_ROOT)

    @staticmethod
    def MRART() -> Self:
        """Method to create object for HBN CUNY site"""
        return BIDSDirectory(config.MRART_FOLDER, root_dir=config.DATASET_ROOT)


def get_sub(path: str):
    req = r".*(sub-[\dA-Za-z]*).*"
    match = re.match(req, path)
    sub = None
    if match is not None:
        groups = match.groups()
        sub = groups[0]
    return sub


def get_ses(path: str):
    req = r".*(ses-[\dA-Za-z]*).*"
    match = re.match(req, path)
    sub = None
    if match is not None:
        groups = match.groups()
        sub = groups[0]
    return sub
