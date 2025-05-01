"""Module to generate synthetic motion data."""

import json
import logging
import os
import shutil
from typing import Any

import pandas as pd
import torch
import torchio as tio
import tqdm
from joblib import Parallel, delayed
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    RandomizableTransform,
    SaveImage,
    ScaleIntensityd,
    Transform,
)
from torchio.transforms import RandomElasticDeformation, RandomFlip

from src import config
from src.process.generate.custom_motion import CustomMotion
from src.utils.bids import BIDSDirectory


class CreateSynthVolume(RandomizableTransform):
    """Transform to produce a synthetic volume.

    It applies :
    - quantified random motion
    - random elastic deformation
    - random flip
    - random gamma
    - random bias field
    """

    apply_motion: bool
    apply_elastic: bool
    apply_flip: bool

    # Synthetic Parameters, not meant to be change often
    motion_prob: float = 0.98
    elastic_prob: float = 0.95
    flip_prob: float = 0.5
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
        goal_motion_range=(0.01, 4),
        motion_prob=1,
        motion_only=False,
    ):
        """Initialize Synthetic volume pipeline.

        Args:
            prob (float, optional): Probability to apply transformation. Defaults to 1.
            do_transform (bool, optional): Used to prevent any tranformation. Defaults to True.
            goal_motion_range (tuple, optional): Possible range of motion. Defaults to (0.01, 3).
            motion_prob (int, optional): Probability to apply motion. Defaults to 1.
            motion_only (bool, optional): Prevent any other transformation. Defaults to False.
        """
        super().__init__(prob, do_transform)
        self.elastic_tsf = RandomElasticDeformation(
            num_control_points=7, max_displacement=8, image_interpolation="bspline"
        )

        self.flip = RandomFlip(0, flip_probability=1)
        self.goal_motion = 0
        self.motion_only = motion_only
        self.goal_motion_range = goal_motion_range
        self.motion_prob = motion_prob

    def get_parameters(
        self,
    ) -> dict[str, int | float | tuple[int | float, int | float]]:
        """Return a dictionnary summarizing all parameters for transformation.

        Returns:
            dict[str, Number | tuple[Number, Number]]: Parameters for synthetic
            generation
        """
        return {
            "motion_prob": self.motion_prob,
            "elastic_prob": self.elastic_prob,
            "flip_prob": self.flip_prob,
            "goal_motion_range": self.goal_motion_range,
            "num_transforms_range": self.num_transforms_range,
            "tolerance": self.tolerance,
            "motion_only": self.motion_only,
        }

    def randomize(self, _=None):
        """Determine wich transform to apply."""
        super().randomize(None)
        self.apply_motion = self.R.rand() <= self.motion_prob
        self.apply_elastic = self.R.rand() <= self.elastic_prob
        self.apply_flip = self.R.rand() <= self.flip_prob
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
        """Apply transform on given data."""
        img: torch.Tensor = data["data"]

        self.randomize()
        if self.apply_flip and not self.motion_only:
            img = self.flip(img)
        if self.apply_elastic and not self.motion_only:
            sub = tio.Subject(data=tio.ScalarImage(tensor=img))
            transformed = self.elastic_tsf(sub)
            img = transformed["data"].data
        clear = img.clone()
        if self.apply_motion:
            img, motion_mm = self.motion_tsf(img)
        else:
            motion_mm = 0

        return {
            "data": img,
            "clear": clear,
            "motion_mm": motion_mm,
            "motion_binary": self.apply_motion,
        }


class Preprocess(Compose):
    """Transform to prepocess MRI volume before synthetic motion generation."""

    def __init__(self):
        """Initialize basic preprocessing."""
        self.tsf = [
            LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
            Orientationd(keys="data", axcodes="RAS"),
            CenterSpatialCropd(keys="data", roi_size=config.VOLUME_SHAPE),
            ScaleIntensityd(keys="data", minv=0, maxv=1),
        ]
        super().__init__(self.tsf)


class SyntheticPipeline(Transform):
    """Transform representing Synthetic generation process until volume storing."""

    def __init__(
        self,
        dataset_dir: BIDSDirectory,
        new_dataset_dir: str,
        n_iterations=config.NUM_ITERATIONS,
        freesurfer_sim=False,
    ):
        """Initialize synthetic pipeline.

        Args:
            dataset_dir (BIDSDirectory): Directory containing base data
            new_dataset_dir (str): Directory to store synthetic data
            n_iterations (_type_, optional): Number of generation to perform on a single volume.
             Defaults to config.NUM_ITERATIONS.
            freesurfer_sim (bool, optional): Flag to specify special FreeSurfer configuration.
             Apply only motion transormation to allow fair thickness comparison. Defaults to False.
        """
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
                goal_motion_range=(0.01, 3), motion_prob=1, motion_only=True
            )
        else:
            self.synthetic_tsf = CreateSynthVolume()
        self.process = Compose(
            [
                self.synthetic_tsf,
            ]
        )

    def save_parameters(self):
        """Save synthetic parameters for reproducibility purpose."""
        params = self.synthetic_tsf.get_parameters()
        with open(
            f"{self.new_dataset_dir}/generation_parameters.json", "w", encoding="utf8"
        ) as file:
            json.dump(params, file)

    def __call__(self, path: str) -> list[dict[str, Any]]:
        """Tranform and store the synthetic volume.

        Args:
            path (str): Element to process

        Returns:
            dict[str, Any]: dict containing
            the data to store as csv
        """
        element = {"data": path}
        volume = self.load(element)
        subject, session = self.dataset_dir.extract_sub_ses(element["data"])

        generated = []

        if self.freesurfer_sim:
            orig_dir_path = os.path.join(
                self.new_dataset_dir, f"{subject}_{session}_gen-orig"
            )
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
            dir_path = os.path.join(self.new_dataset_dir, f"{subject}_{session}_{gen}")

            new_identifier = f"{subject}_{session}_{gen}"
            corrupted_path = os.path.join(dir_path, f"{new_identifier}_corrupted_T1w")

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
            generated.append(sample)
        return generated


def launch_generate_data(
    dataset_dir: BIDSDirectory,
    new_dataset: str,
    root_dir: str,
    num_iteration: int,
    freesurfer_sim=False,
):
    """Generate synthetic motion dataset and store everything (Volumes and CSVs).

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
        )(dataset_dir.get_t1w(*sub_ses))
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
