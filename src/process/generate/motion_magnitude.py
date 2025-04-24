"""
This code comes from : https://github.com/Deep-MI/head-motion-tools/tree/main
Presented in the article : Pollak, C., Kügler, D., Breteler, M.M. and Reuter, M., 2023.
Quantifying MR head motion in the Rhineland Study–A robust method 
for population cohorts. NeuroImage, 275, p.120176.
"""

import warnings

import numpy as np
from numba import njit
from numba.core.errors import NumbaPerformanceWarning
from scipy.spatial.transform import Rotation
from torchio import Motion

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def get_affine(rot: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """Create affine matrix from rotation and translation vector

    Args:
        rot (np.ndarray): rotation vector
        transl (np.ndarray): translation vector

    Returns:
        np.ndarray: affine matrix
    """
    rot_mat = Rotation.from_rotvec(rot, True).as_matrix()
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rot_mat
    affine_matrix[:3, 3] = transl
    return affine_matrix


def get_matrices(transf_hist: Motion) -> np.ndarray:
    """Return affine matrices from an applied motion transform

    Args:
        transf_hist (Motion): Motion tranform used to modify a volume

    Returns:
        np.ndarray: affine matrices
    """
    rotations = transf_hist.degrees["data"]
    translations = transf_hist.translation["data"]
    affine_matrices = [np.eye(4)]
    for rot, transl in zip(rotations, translations):
        affine_matrices.append(get_affine(rot, transl))
    return affine_matrices


def get_motion_dist(affine_matrices: np.ndarray) -> float:
    """Compute average motion using the code from Pollak, C. et al.

    Args:
        affine_matrices (np.ndarray): list of successive affine matrices

    Returns:
        float: motion quantification
    """
    dist = quantify_motion(affine_matrices)
    return np.array(dist).mean()


@njit()
def ms_dev(A, B=np.identity(4), x=np.zeros((1, 3)), r=80):
    """
    Calculates the root mean square deviation of two homogenous transformations in 3d
    This distance is used in Jenkinson 1999 RMS deviation - tech report
    www.fmrib.ox.ac.uk/analysis/techrep .

    A       homogenous transformation matrix
    B       homogenous transformation matrix (identity by default)
    x       sphere center (brain center, can be RAS center)
    r       sphere radius (head size)
    return  the root mean square deviation
    """
    assert x.shape[0] == 1, "x must be a 1x3 array"
    assert x.shape[1] == 3, "x must be a 1x3 array"

    A = B @ np.linalg.inv(A) - np.identity(4)
    t = np.expand_dims(A[:3, 3], 0).T
    A = A[:3, :3]

    ret = (1 / 5) * (r**2) * np.trace(A.T @ A) + (t + A @ x.T).T @ (t + A @ x.T)
    return ret.item()


@njit()
def rms_dev(A, B=np.identity(4), x=np.zeros((1, 3)), r=80):
    return np.sqrt(ms_dev(A, B, x, r))


def quantify_motion(transformation_series, head_center=np.zeros((1, 3)), seq="T1"):
    """
    Calculates the speed of movement from transformations.

    Parameters:
    - transformation_series: numpy array of homogenous transformations
    - head_center: numpy array representing the center of the head (default: 3x3 array of zeros)
    - seq: MRI sequence to analyze (default: 'T1')
    - correct_timestamps: boolean indicating whether to correct timestamps (default: True)

    Returns:
    - Speed of movement as calculated by the quantifier function.
    """
    return quantifier(
        transformation_series,
        mode="speed",
        head_center=head_center,
        from_starting_position=False,
        seq=seq,
    )


def quantify_deviation(
    transformation_series,
    head_center=np.zeros((1, 3)),
    zero_in=True,
    seq="T1",
    mode="RMSD",
):
    """
    wrapper for "quantifier"
    calculates the distance to starting point

    Parameters:
    transformation_series: numpy array of homogenous transformations
    head_center: numpy array representing the center of the head
        (default: 3x3 array of zeros)
    zeroIn: boolean indicating whether to calculate the distance
        from the starting point (default: True)
    seq: MRI sequence to analyze (default: 'T1')
    mode: string indicating the mode to use (default: 'RMSD')
    correct_timestamps: boolean indicating whether to correct
        timestamps (default: True)
    """
    if not (mode == "RMSD" or mode == "centroid"):
        raise ValueError("Wrong mode identifier")
    return quantifier(
        transformation_series,
        mode=mode,
        head_center=head_center,
        from_starting_position=zero_in,
        seq=seq,
    )


def quantifier(
    transforms,
    mode,
    head_center=np.zeros((1, 3)),
    from_starting_position=True,
    seq="FULL",
):
    """
    calculates different quantifiers for the input transformations
    see wrapper functions quantifyDeviation, quantifyMotion

    transform_dict      dictionary of with lists of transformations as numpy arrays
    mode                (RMSD|centroid|speed) different measures
    sr                  sphere radius for root mean square distance
    from_starting_position when True the distances will be calculated with respect to the first transformation in the set
                            otherwise just the size of the given transforms - only applies to mode RMSD
    smoothing_dist      how many values to include in the smoothing
    seq                 MRI sequence to analyze
    mode                (RMSD|centroid|speed) different measures

    return  output          containing the quantifier as specified in 'mode'
            avgs            average smoothed output
            mad             median absolute deviation of smoothed RMSD_diff_dict
    """
    if not (mode == "RMSD" or mode == "centroid" or mode == "speed"):
        raise ValueError("Wrong mode identifier")

    if from_starting_position is not False:
        if type(from_starting_position).__module__ == np.__name__:
            starting_trans = from_starting_position
            from_starting_position = True

    if head_center is None:
        print("head center unknown")
        RMS_r = 82.5
        RMS_x = np.array([[-7.89449484, -48.07730192, 239.47502511]])
    else:
        RMS_r = 82.5
        RMS_x = head_center

    arr = np.array([])
    if mode == "RMSD":
        if from_starting_position:
            starting_trans = transforms[0]
            if np.isnan(starting_trans).any():
                trans_arr = np.array(transforms)
                trans_arr = trans_arr[~np.isnan(np.array(transforms)[:, 0, 0])]
                starting_trans = trans_arr[0]

        RMSDs = []
        for i, _ in enumerate(transforms):
            if np.isnan(transforms[i]).any():  # enforce numpy nan handling for numba
                RMSDs.append(np.nan)
            else:
                if from_starting_position:
                    RMSDs.append(
                        rms_dev(starting_trans, transforms[i], r=RMS_r, x=RMS_x)
                    )
                else:
                    RMSDs.append(rms_dev(transforms[i], r=RMS_r, x=RMS_x))

        arr = np.array(RMSDs)

    elif mode == "speed":
        RMSD_diffs = []
        prev = transforms[0]
        for i, _ in enumerate(transforms):
            if (
                np.isnan(transforms[i]).any() or np.isnan(prev).any()
            ):  # enforce numpy nan handling for numba
                RMSD_diffs.append(np.nan)
            else:
                RMSD_diffs.append(rms_dev(prev, transforms[i], r=RMS_r, x=RMS_x))
            prev = transforms[i]

        arr = np.array(RMSD_diffs)

    return arr.tolist()
