"""Module to generate synthetic motion data"""

import json
import logging
import os
import shutil
from numbers import Number
from typing import Callable

import pandas as pd
import torch
import tqdm
from joblib import Parallel, delayed
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    RandomizableTransform,
    SaveImage,
    ScaleIntensityd,
    Transform,
)
from torchio.transforms import (
    RandomBiasField,
    RandomElasticDeformation,
    RandomFlip,
    RandomGamma,
)

from src import config
from src.process.generate.custom_motion import CustomMotion
from src.utils.bids import BIDSDirectory


class CreateSynthVolume(RandomizableTransform):
    """Transform to produce a synthetic volume using:
    - quantified random motion
    - random elastic deformation
    - random flip
    - random gamma
    - random bias field
    """

    apply_motion: bool
    apply_elastic: bool
    apply_flip: bool
    apply_corrupt: bool

    # Synthetic Parameters, not meant to be change often
    motion_prob: float = 0.98
    elastic_prob: float = 0.9
    flip_prob: float = 0.5
    corrupt_prob: float = 0.3
    goal_motion_range: tuple[float, float]
    num_transforms_range: tuple[int, int] = (2, 8)
    tolerance: float = 0.02

    motion_tsf: CustomMotion
    goal_motion: float
    num_transforms: int

    def __init__(
        self,
        prob: float = 1,
        do_transform: bool = True,
        goal_motion_range=(0.01, 2.0),
        motion_prob=0.98,
        motion_only=False,
    ):
        super().__init__(prob, do_transform)
        self.elastic_tsf = RandomElasticDeformation(
            num_control_points=7, max_displacement=8, image_interpolation="bspline"
        )

        self.corrupt = Compose(
            [RandomGamma((-0.05, 0.05)), RandomBiasField(coefficients=(0.0, 0.2))]
        )
        self.flip = RandomFlip(0, flip_probability=1)
        self.goal_motion = 0
        self.motion_only = motion_only
        self.goal_motion_range = goal_motion_range
        self.motion_prob = motion_prob

    def get_parameters(self) -> dict[str, Number | tuple[Number, Number]]:
        """Return a dictionnary summarizing all parameters for transformation

        Returns:
            dict[str, Number | tuple[Number, Number]]: Parameters for synthetic
            generation
        """
        return {
            "motion_prob": self.motion_prob,
            "elastic_prob": self.elastic_prob,
            "flip_prob": self.flip_prob,
            "corrupt_prob": self.corrupt_prob,
            "goal_motion_range": self.goal_motion_range,
            "num_transforms_range": self.num_transforms_range,
            "tolerance": self.tolerance,
            "motion_only": self.motion_only,
        }

    def randomize(self):
        """Determine wich transform to apply"""
        super().randomize(None)
        self.apply_motion = self.R.rand() <= self.motion_prob
        self.apply_elastic = self.R.rand() <= self.elastic_prob
        self.apply_flip = self.R.rand() <= self.flip_prob
        self.apply_corrupt = self.R.rand() <= self.corrupt_prob
        if self.apply_motion:
            self.goal_motion = self.R.uniform(*self.goal_motion_range)

            self.motion_tsf = CustomMotion(
                self.goal_motion,
                tolerance=self.tolerance,
                num_transforms_range=self.num_transforms_range,
            )
        else:
            self.num_transforms = 0

    def __call__(self, data):
        img: torch.Tensor = data["data"]
        self.randomize()
        if self.apply_flip and not self.motion_only:
            img = self.flip(img)
        if self.apply_elastic and not self.motion_only:
            img = self.elastic_tsf(img)
        clear = img.clone()
        if self.apply_motion:
            img, motion_mm = self.motion_tsf(img)
        else:
            motion_mm = 0

        if self.apply_corrupt and not self.motion_only:
            img = self.corrupt(img)

        return {
            "data": img,
            "clear": clear,
            "motion_mm": motion_mm,
            "motion_binary": self.apply_motion,
        }


class Preprocess(Compose):
    """Transform to prepocess MRI volume before synthetic motion generation"""

    def __init__(self):
        self.tsf = [
            LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
            Orientationd(keys="data", axcodes="RAS"),
            ScaleIntensityd(keys="data", minv=0, maxv=1),
        ]
        super().__init__(self.tsf)


class RandomScaleIntensityd(RandomizableTransform):
    """Transform to randomly scale intensity from min to
    max_range drawn with uniform distribution"""

    scaler: Callable

    def __init__(self, keys="data", minv=0, max_range=config.INTENSITY_SCALING):
        """
        Args:
            keys (str, optional): keys to apply transform to. Defaults to "data".
            minv (int, optional): min value for rescale. Defaults to 0.
            max_range (tuple, optional): max value range for rescale. Defaults to (0.9, 1.1).
        """
        super().__init__(prob=1)
        self.max_range = max_range
        self.minv = minv
        self.keys = keys

    def randomize(self):
        """Create the scale transform with random max (uniform distribution)"""
        self.scaler = ScaleIntensityd(
            keys=self.keys, minv=self.minv, maxv=self.R.uniform(*self.max_range)
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        self.randomize()
        return self.scaler(data)


class SyntheticPipeline(Transform):
    """Transform representing Synthetic generation process until volume storing"""

    def __init__(
        self,
        dataset_dir: BIDSDirectory,
        new_dataset_dir: str,
        n_iterations=config.NUM_ITERATIONS,
        freesurfer_sim=False,
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.dataset_dir = dataset_dir
        self.new_dataset_dir = new_dataset_dir
        self.freesurfer_sim = freesurfer_sim
        self.load = Preprocess()
        self.save = SaveImage(
            savepath_in_metadict=True,
            resample=False,
            separate_folder=False,
            print_log=False,
        )
        if self.freesurfer_sim:
            self.synthetic_tsf = CreateSynthVolume(
                goal_motion_range=(0.01, 2.5), motion_prob=1, motion_only=True
            )
        else:
            self.synthetic_tsf = CreateSynthVolume()
        self.process = Compose(
            [
                self.synthetic_tsf,
                RandomScaleIntensityd(
                    keys="data", minv=0, max_range=config.INTENSITY_SCALING
                ),
            ]
        )

    def __call__(self, path: str) -> dict[str, int | float | str | bool]:
        """Tranform and store the synthetic volume,
        returns all metadata to store as a dict

        Args:
            path (str): Element to process

        Returns:
            dict[str, int | float | str | bool]: dict containing
            the data to store as csv
        """
        element = {"data": path}
        volume = self.load(element)
        subject, session = self.dataset_dir.extract_sub_ses(element["data"])
        sub_ses_path = os.path.join(self.new_dataset_dir, subject, session)

        generated = []

        if self.freesurfer_sim:
            orig_dir_path = os.path.join(sub_ses_path, "gen-orig", "anat")
            os.makedirs(orig_dir_path, exist_ok=True)
            new_identifier = f"{subject}_{session}_gen-orig"
            orig_file_path = os.path.join(orig_dir_path, f"{new_identifier}_T1w.nii.gz")
            shutil.copy(path, orig_file_path)
            generated.append(
                {
                    "subject": subject,
                    "session": session,
                    "generation": "gen-orig",
                    "motion_mm": 0,
                    "motion_binary": False,
                    "identifier": new_identifier,
                    "data": orig_file_path,
                }
            )

        for curr_iter in range(self.n_iterations):
            synth = self.process(volume)

            gen = f"gen-{str(curr_iter).zfill(3)}"
            dir_path = os.path.join(sub_ses_path, gen, "anat")

            new_identifier = f"{subject}_{session}_{gen}"
            corrupted_path = os.path.join(dir_path, f"{new_identifier}_corrupted_T1w")
            clear_path = os.path.join(dir_path, f"{new_identifier}_clear_T1w")

            os.makedirs(dir_path, exist_ok=True)

            self.save(synth["data"], filename=corrupted_path)

            corrupt_rel_path = os.path.relpath(
                corrupted_path, os.path.dirname(self.new_dataset_dir)
            )
            sample = {
                "subject": subject,
                "session": session,
                "generation": gen,
                "motion_mm": synth["motion_mm"],
                "motion_binary": synth["motion_binary"],
                "identifier": new_identifier,
                "data": corrupt_rel_path + ".nii.gz",
            }
            if not self.freesurfer_sim:
                self.save(synth["clear"], filename=clear_path)
                clear_rel_path = os.path.relpath(
                    clear_path, os.path.dirname(self.new_dataset_dir)
                )
                sample["clear"] = (clear_rel_path + ".nii.gz",)
            generated.append(sample)
        return generated

    def save_parameters(self):
        """Save synthetic parameters for reproducibility purpose"""
        params = self.synthetic_tsf.get_parameters()
        with open(
            f"{self.new_dataset_dir}/generation_parameters.json", "w", encoding="utf8"
        ) as file:
            json.dump(params, file)


def launch_generate_data(
    dataset_dir: BIDSDirectory,
    new_dataset: str,
    root_dir: str,
    num_iteration: int,
    freesurfer_sim=False,
):
    """Generate synthetic motion dataset and store everything (Volumes and CSVs)

    Args:
        new_dataset (str): Name of the new dataset
    """
    new_dataset_dir = os.path.join(root_dir, new_dataset)
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)

    logging.info("[SYNTH DATASET] Create synth dataset in %s", new_dataset_dir)

    dataset_paths = list(dataset_dir.walk())
    if freesurfer_sim:
        dataset_paths = dataset_paths[: config.FREESURFER_NUM_SUBJECTS]
    dataset_length = len(dataset_paths)
    logging.info(
        "[SYNTH DATASET] Starting : %d iteration per volumes for %d volumes",
        num_iteration,
        dataset_length,
    )
    logging.info("[SYNTH DATASET] Will create %d", num_iteration * dataset_length)

    generated_all = []
    parallel = Parallel(
        n_jobs=config.NUM_PROCS,
        return_as="generator_unordered",
    )(
        delayed(
            SyntheticPipeline(
                dataset_dir, new_dataset_dir, num_iteration, freesurfer_sim
            )
        )(dataset_dir.get_T1w(*sub_ses))
        for sub_ses in dataset_paths
    )

    for sub_generated in tqdm.tqdm(parallel, total=dataset_length):
        generated_all += sub_generated

    pd.DataFrame.from_records(generated_all).to_csv(f"{new_dataset_dir}/scores.csv")

    synth_tsf = SyntheticPipeline(
        dataset_dir, new_dataset_dir, num_iteration, freesurfer_sim
    )
    synth_tsf.save_parameters()

    logging.info("[SYNTH DATASET] DONE !")
