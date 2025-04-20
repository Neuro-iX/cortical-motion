import glob
import os
import re
from typing import Generator, Self

from src import config


class BIDSDirectory:
    """Utility to query BIDS directories"""

    has_session: bool = False
    has_sites: bool = False
    sites: list[str] = None

    def __init__(
        self, dataset: str, root_dir: str = config.DATASET_ROOT, has_gen=False
    ):
        """Args:
        dataset (str): Dataset directory name
        root_dir (str): Root directory for all datasets
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.base_path = os.path.join(self.root_dir, self.dataset)
        self.has_gen = has_gen
        self.has_session = len(glob.glob("sub-*/ses-*", root_dir=self.base_path)) > 0

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

    def get_gen(self, sub_id: str, ses_id: str) -> list[str]:
        """Retrieve list of seesion
        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str): Session id (ses-???)

        Returns:
            list[str]: list of session
        """

        return glob.glob("gen-*", root_dir=os.path.join(self.base_path, sub_id, ses_id))

    def walk(self) -> Generator[tuple[str, str], None, None]:
        """Iterate over every subjects and session of the dataset
        Yields:
            Generator[tuple[str, str], None, None]: sub_id, ses_id
        """
        for sub in self.get_subjects():
            if self.has_session:
                for ses in self.get_session(sub):
                    if self.has_gen:
                        for gen in self.get_gen(sub, ses):
                            yield sub, ses, gen
                    else:
                        yield sub, ses
            else:

                yield sub, None

    def __len__(self):
        return len(list(self.walk()))

    def get_T1w(self, sub_id: str, ses_id: str = "ses-1", gen_id=None) -> str:
        """Return path to T1w volume

        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str, optional): Session id (ses-???). Defaults to "ses-1".

        Returns:
            str: Path to T1w
        """
        if gen_id is not None:
            return glob.glob(
                os.path.join(
                    self.base_path, sub_id, ses_id, gen_id, "anat", "*T1w.nii*"
                )
            )[0]
        else:
            if ses_id is not None:
                return glob.glob(
                    os.path.join(self.base_path, sub_id, ses_id, "anat", "*T1w.nii*")
                )[0]
            else:
                return glob.glob(
                    os.path.join(self.base_path, sub_id, "anat", "*T1w.nii*")
                )[0]

    def get_all_T1w(self, sub_id: str, ses_id: str = "ses-1", gen_id=None) -> str:
        """Return path to T1w volume

        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str, optional): Session id (ses-???). Defaults to "ses-1".

        Returns:
            str: Path to T1w
        """
        if gen_id is not None:
            return glob.glob(
                os.path.join(
                    self.base_path, sub_id, ses_id, gen_id, "anat", "*T1w.nii.gz"
                )
            )
        else:
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
    def FS_SYNTH() -> Self:
        """Method to create object for FreeSurfer's analysis dataset"""
        return BIDSDirectory(
            config.FREESURFER_SYNTH_FOLDER, root_dir=config.DATASET_ROOT, has_gen=True
        )


class ClinicaDirectory(BIDSDirectory):
    def __init__(self, dataset: str, root_dir: str = config.DATASET_ROOT):
        super().__init__(dataset, root_dir)
        self.base_path = os.path.join(self.base_path, "subjects")
        self.has_session = len(glob.glob("sub-*/ses-*", root_dir=self.base_path)) > 0

        print(self.base_path)

    def get_all_T1w(self, sub_id: str, ses_id: str = "ses-1") -> str:
        """Return path to T1w volume

        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str, optional): Session id (ses-???). Defaults to "ses-1".

        Returns:
            str: Path to T1w
        """
        return glob.glob(
            os.path.join(
                self.base_path,
                sub_id,
                ses_id,
                "t1_linear",
                "*space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz",
            )
        )

    def get_T1w(self, sub_id: str, ses_id: str = "ses-1") -> str:
        """Return path to T1w volume

        Args:
            sub_id (str): Subject id (sub-???)
            ses_id (str, optional): Session id (ses-???). Defaults to "ses-1".

        Returns:
            str: Path to T1w
        """
        return glob.glob(
            os.path.join(
                self.base_path,
                sub_id,
                ses_id,
                "t1_linear",
                "*space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz",
            )
        )[0]

    @staticmethod
    def CBICCUNY() -> Self:
        """Method to create object for HBN CUNY site"""
        return ClinicaDirectory(config.CBICCUNY_FOLDER, root_dir=config.DATASET_ROOT)


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
