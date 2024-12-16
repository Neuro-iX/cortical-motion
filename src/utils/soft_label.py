from __future__ import annotations

from typing import Dict, Hashable, Mapping, Union

import numpy as np
import torch
from monai.transforms import MapTransform
from scipy.stats import norm

from src import config


class ToSoftLabel(MapTransform):
    """
    Utility transform to use soft labelling as define in :
    Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021).
    Accurate brain age prediction with lightweight deep neural networks.
    Medical Image Analysis, 68, 101871.
    https://doi.org/10.1016/j.media.2020.101871
    """

    def __init__(
        self,
        keys,
        backup_keys,
        bin_range: tuple,
        bin_step: float,
        soft_label: bool = True,
        require_grad=False,
    ):
        """
        Adapted from :
        https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/dp_model/dp_utils.py

        v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
        bin_range: (start, end), size-2 tuple
        bin_step: should be a divisor of |end-start|
        soft_label:True 'soft label', v is vector else 'hard label', v is index
        debug: True for error messages.
        """
        super().__init__(keys)
        if isinstance(backup_keys, tuple):
            self.backup_keys = backup_keys
        else:
            self.backup_keys = (backup_keys,)

        self.bin_start = bin_range[0]
        self.bin_end = bin_range[1]
        self.bin_length = self.bin_end - self.bin_start
        if not round(self.bin_length / bin_step, 5) % 1 == 0:
            raise ValueError("bin's range should be divisible by bin_step!")

        # self.bin_range = bin_range
        self.bin_step = bin_step
        self.soft_label = soft_label
        self.bin_number = int(round(self.bin_length / bin_step))
        self.bin_centers = (
            self.bin_start + float(bin_step) / 2 + bin_step * np.arange(self.bin_number)
        )

        if require_grad:
            self.bin_centers = torch.tensor(self.bin_centers, dtype=torch.float32)

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor]]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, backup in zip(self.keys, self.backup_keys):
            if torch.is_tensor(d[key]):
                d[backup] = d[key].clone()
            else:
                d[backup] = d[key]

            d[key] = self.value_to_softlabel(d[key])

        return d

    def _get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Utility function to retrieve probability vector from input
        If sum of every element is more than 1, we consider that the input
        vector need a log_softmax

        Args:
            x (torch.Tensor): Input vector either :
              - raw output
              - log_softmax processed output

        Returns:
            torch.Tensor: _description_
        """
        if torch.sum(x) > 1.01:
            x = torch.nn.functional.log_softmax(x.squeeze())

        return torch.exp(x)

    def value_to_softlabel(
        self, x: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Convert a vector of single values to a vector of soft label

        Args:
            x (torch.Tensor | np.array): Vector of label as Tensor or array

        Returns:
            torch.Tensor |np.array: Vector of softlabel in the input type
        """
        if torch.is_tensor(x):
            was_tensor = True
            x = x.squeeze().numpy()
            assert len(x.shape) == 1 or len(x.shape) == 0
            x = x.tolist()
        else:
            was_tensor = False

        if not self.soft_label:
            x = np.array(x)
            i = np.floor((x - self.bin_start) / self.bin_step)
            i = i.astype(int)
            return i if not was_tensor else torch.tensor(i)
        else:
            if np.isscalar(x):
                v = np.zeros((self.bin_number,))
                for i in range(self.bin_number):
                    x1 = self.bin_centers[i] - float(self.bin_step) / 2
                    x2 = self.bin_centers[i] + float(self.bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x, scale=self.bin_length * 0.03)
                    v[i] = cdfs[1] - cdfs[0]
            else:
                v = np.zeros((len(x), self.bin_number))
                for j in range(len(x)):
                    for i in range(self.bin_number):
                        x1 = self.bin_centers[i] - float(self.bin_step) / 2
                        x2 = self.bin_centers[i] + float(self.bin_step) / 2
                        cdfs = norm.cdf(
                            [x1, x2], loc=x[j], scale=self.bin_length * 0.03
                        )
                        v[j, i] = cdfs[1] - cdfs[0]

            return v if not was_tensor else torch.tensor(v)

    def logsoft_to_hardlabel(self, x: torch.Tensor) -> torch.Tensor:
        """Convert soft label (in log format) to hard label

        Args:
            x (torch.Tensor): Vector of soft label or single soft label

        Returns:
            torch.Tensor: Vector or single hard label
        """
        if x.get_device() != -1:
            # if not on CPU
            x = x.cpu()
        pred = self._get_probs(x) @ self.bin_centers

        return torch.as_tensor(pred)

    @staticmethod
    def motion_config() -> ToSoftLabel:
        """Create an instance with basic configuration (see config.py)

        Returns:
            Self: Instance with basic config
        """
        return ToSoftLabel(
            keys="label",
            backup_keys="motion_mm",
            bin_range=config.MOTION_BIN_RANGE,
            bin_step=config.MOTION_BIN_STEP,
        )
