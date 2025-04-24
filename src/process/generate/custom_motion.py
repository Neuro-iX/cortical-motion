"""Module containing motion related transforms."""

from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
import torchio as tio
from monai.transforms import RandomizableTransform

from src.process.generate import motion_magnitude


class CustomMotion(tio.transforms.RandomMotion, RandomizableTransform):
    """
    Adaptation of torchIO RandomMotion to generate volume with a goal quantified motion.

    We use it to have a more uniform label distribution in the synthetic dataset.
    """

    def __init__(
        self,
        goal_motion: float,
        tolerance: float = 0.02,
        num_transforms_range: tuple[int, int] = (2, 8),
    ):
        """Randomly generate a motion in the range [goal_motion-tolerance, goal_motion+tolerance].

        Args:
            goal_motion (float): quantify motion wanted
            tolerance (float, optional): acceptable tolerance. Defaults to 0.02.
        """
        self.transform_degrees = self.R.uniform(0, np.max((goal_motion * 2, 1)))
        self.transform_translation = self.R.uniform(0, np.max((goal_motion, 1)))
        self.goal_motion = goal_motion
        self.num_transforms = self.R.randint(
            low=num_transforms_range[0], high=num_transforms_range[1]
        )
        self.tolerance = tolerance

        super().__init__(
            self.transform_degrees,
            self.transform_translation,
            self.num_transforms,
            image_interpolation="bspline",
            parse_input=False,
        )

    def randomize(self, data: Any = None):
        """Randomize degree and translations range.

        Args:
            data (Any): Not Used (Defaults to None)
        """
        self.degrees_range = self.parse_degrees(
            self.R.uniform(0, np.max((self.goal_motion * 2, 1)))
        )
        self.translation_range = self.parse_translation(
            self.R.uniform(0, np.max((self.goal_motion, 1)))
        )

    def apply_transform(self, image: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Apply motion simulation.

        Same transformation as torchIO RandomMotion, retry until goal motion is produced.

        Args:
            image (torch.Tensor): Volume tensor

        Returns:
            tuple[torch.Tensor, float]: Subject with modified volumes
        """
        arguments: Dict[str, dict] = defaultdict(dict)
        motion_mm = -1
        self.randomize(0)
        retry = 0
        while (
            motion_mm > self.goal_motion + self.tolerance
            or motion_mm < self.goal_motion - self.tolerance
        ):
            params = self.get_params(
                self.degrees_range,
                self.translation_range,
                self.num_transforms,
                is_2d=False,
            )
            times_params, degrees_params, translation_params = params
            rotations = degrees_params
            translations = translation_params
            affine_matrices = [np.eye(4)]
            for rot, transl in zip(rotations, translations):
                affine_matrices.append(motion_magnitude.get_affine(rot, transl))
            motion_mm = motion_magnitude.get_motion_dist(affine_matrices)
            retry += 1
            if retry > 10:
                self.randomize(0)
                retry = 0
        sub = tio.Subject(data=tio.ScalarImage(tensor=image))
        name = "data"
        arguments["times"][name] = times_params
        arguments["degrees"][name] = degrees_params
        arguments["translation"][name] = translation_params
        arguments["image_interpolation"][name] = self.image_interpolation
        transform = tio.transforms.Motion(**self.add_include_exclude(arguments))
        transformed = transform(sub)
        assert isinstance(transformed, tio.Subject)
        return transformed["data"].data, motion_mm
