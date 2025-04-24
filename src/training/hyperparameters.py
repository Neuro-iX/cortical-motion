"""Module defining Hyperparameter class, enumeration and search functions"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.utils.networks import get_hyperparams


class TuningTask(Enum):
    """Possible tuning objectives / task"""

    CONV = "conv"
    DOWN = "down"
    NORM = "norm"
    ACT = "act"
    BINS = "bins"
    ADAM = "adam"
    REG = "regression"
    CONTRAST = "contrast"
    ABLATION_ARCHI = "ablation-architecture"
    ABLATION_AUG = "ablation-augmentation"
    ABLATION_LOSS = "ablation-loss"


class DownsampleType(Enum):
    """Pooling strategies"""

    POOL = "pool"
    STRIDE = "stride"
    AUG_STRIDE = "augmented_stride"
    NONE = "none"


class NormType(Enum):
    """Normalization strategies"""

    BATCH = "batch"
    INSTANCE = "instance"
    LAYER = "layer"
    NONE = "none"
    GROUP = "group"


class ActivationType(Enum):
    """Activation functions"""

    RELU = "relu"
    PRELU = "prelu"


class RegressionLossType(Enum):
    """Possible loss functions"""

    KL_DIV = "kldiv"
    L2 = "l2"
    L1 = "l1"
    MIXED = "mixed"


class ClassifierType(Enum):
    """Type of classifier"""

    SFCN = "sfcn"
    SFCN_LONG = "sfcn_long"
    VANILLA_REG = "vanilla_reg"


@dataclass
class HyperParamConf:
    """Class storing all hyperparameters configuration"""

    idx: int
    task: TuningTask = TuningTask.CONV

    batch_size: int = 10
    kernel_size: int = 3
    channels: Sequence[int] = (1, 32, 64, 128, 256, 256, 64)

    n_convs: Sequence[int] = (2, 2, 2, 2, 2, 2)

    norm: NormType = NormType.BATCH

    down: DownsampleType = DownsampleType.POOL

    act: ActivationType = ActivationType.RELU

    n_bins: int = 50
    kl_beta: float = 1

    loss: RegressionLossType = RegressionLossType.KL_DIV

    classifier: ClassifierType = ClassifierType.SFCN

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    lr: float = 1e-3
    fixed_lr: bool = True
    dropout: float = 0.6

    augmentation: bool = True
    hist_shift: bool = True

    soft_label_func: str = "norm"
    weighted_loss: bool = False


def sample_conv() -> list[HyperParamConf]:
    """Sample hyperparameter configs to test convolution"""
    hp_list = []

    hp_list = [
        HyperParamConf(idx=0, n_convs=(1, 1, 1, 1, 1, 1)),
        HyperParamConf(idx=1, n_convs=(2, 2, 2, 2, 2, 2)),
        HyperParamConf(idx=2, n_convs=(3, 3, 3, 3, 3, 3)),
        HyperParamConf(idx=3, n_convs=(3, 3, 3, 3, 4, 4)),
    ]
    return hp_list


def sample_down() -> list[HyperParamConf]:
    """Sample hyperparameter configs to test downsampling"""
    hp_list = []
    hyperparam_grid = {
        "down": [DownsampleType.POOL, DownsampleType.STRIDE, DownsampleType.AUG_STRIDE],
    }
    conf_comb = get_hyperparams(hyperparam_grid)
    for i, conf in enumerate(conf_comb):
        hp_list.append(HyperParamConf(idx=i, down=conf["down"]))
    return hp_list


def sample_norm() -> list[HyperParamConf]:
    """Sample hyperparameter configs to test normalization"""
    hp_list = []
    hyperparam_grid = {
        "norm": [NormType.NONE, NormType.BATCH, NormType.INSTANCE],
    }
    conf_comb = get_hyperparams(hyperparam_grid)
    for i, conf in enumerate(conf_comb):
        hp_list.append(HyperParamConf(idx=i, norm=conf["norm"]))
    return hp_list


def sample_act() -> list[HyperParamConf]:
    """Sample hyperparameter configs to test activations"""
    hp_list = []

    hyperparam_grid = {
        "act": [ActivationType.PRELU, ActivationType.RELU],
    }
    conf_comb = get_hyperparams(hyperparam_grid)
    for i, conf in enumerate(conf_comb):
        hp_list.append(HyperParamConf(idx=i, act=conf["act"]))
    return hp_list


def sample_bins_loss() -> list[HyperParamConf]:
    """Sample hyperparameter configs to test losses
    for soft label strategy"""
    hp_list = []

    hyperparam_grid = {
        "n_bins": [30, 40, 50],
        "loss": [
            (RegressionLossType.KL_DIV, 0),
            *[(RegressionLossType.MIXED, beta) for beta in np.linspace(0.1, 1, 3)],
        ],
        "classifier": [ClassifierType.SFCN, ClassifierType.SFCN_LONG],
    }
    conf_comb = get_hyperparams(hyperparam_grid)
    for idx, conf in enumerate(conf_comb):
        hp_list.append(
            HyperParamConf(
                idx=idx,
                n_bins=conf["n_bins"],
                loss=conf["loss"][0],
                kl_beta=conf["loss"][1],
                classifier=conf["classifier"],
            )
        )
    return hp_list


def sample_vanillareg() -> list[HyperParamConf]:
    """Sample hyperparameter configs to test classical regression"""

    return [
        HyperParamConf(
            idx=0,
            loss=RegressionLossType.L2,
            classifier=ClassifierType.VANILLA_REG,
        )
    ]


SAMPLE_TUNING = {
    TuningTask.CONV: sample_conv,
    TuningTask.ACT: sample_act,
    TuningTask.NORM: sample_norm,
    TuningTask.BINS: sample_bins_loss,
    TuningTask.DOWN: sample_down,
    TuningTask.REG: sample_vanillareg,
}
