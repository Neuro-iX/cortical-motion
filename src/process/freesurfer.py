"""Module containing the logic to automate freesurfer jobs on Narval."""

import abc
import logging
import os
import shutil

from simple_slurm import Slurm

from src import config
from src.utils.bids import BIDSDirectory


class Command(abc.ABC):
    """Basic Command wrapper definition."""

    cmd: str

    def compile(self) -> str:
        """Create the final command to execute.

        Returns:
            str: Final command
        """
        return self.cmd


class ApptainerEnv(Command):
    """Command object to use apptainer wrapped commands."""

    def __init__(self, sif_file: str):
        """Initialize an Apptainer environment.

        Args:
        sif_file (str): Path to SIF container
        """
        self.sif_file = sif_file
        self.bind_pairs: list[tuple[str, str]] = []
        self.commands: list[str] = []
        self.cmd = "apptainer run "

    @staticmethod
    def narval_freesurfer() -> "ApptainerEnv":
        """Use config.PATH_FREESURFER_NARVAL.

        Returns:
            ApptainerEnv: Environement for free surfer container
        """
        assert config.PATH_FREESURFER_NARVAL is not None, (
            "No path to freesurfer defined"
        )
        return ApptainerEnv(config.PATH_FREESURFER_NARVAL)

    def bind(self, orig: str, mnt: str):
        """Bind folders to apptainer env.

        Args:
            orig (str): Directory path in normal environment
            mnt (str): Directory path to use in apptainer
        """
        self.bind_pairs.append((orig, mnt))

    def add_command(self, command: str):
        """Add a command to run in apptainer (FIFO).

        Args:
            command (str): commands to add
        """
        self.commands.append(command)

    def compile(self) -> str:
        """Create the final command to execute.

        Returns:
            str: Final command
        """
        cmd = self.cmd
        for pair in self.bind_pairs:
            cmd += f"--bind {pair[0]}:{pair[1]} "
        cmd += self.sif_file
        cmd += ' bash -c "'
        for command in self.commands:
            cmd += f"{command};"
        cmd += '"'
        logging.info(cmd)
        return cmd


class FreeSurferReconAll(Command):
    """Object to wrap FreeSurfer's recon-all command."""

    def __init__(self, subject: str, path: str, subject_dir: str | None = None):
        """Initialize FreeSurfer recon-all command.

        Args:
        subject (str): Subject directory output name
        path (str): Path to the subject T1w volume
        subject_dir (str, optional): Directory for outputs. Defaults to None.
        """
        self.cmd = f"recon-all -all -s {subject} -i {path}"
        if subject_dir:
            self.cmd += f" -sd {subject_dir}"


def run_freesurfer_cortical_thichness(
    sub_id: str, ses_id: str, dataset: BIDSDirectory, gen_id: str | None = None
) -> int:
    """Launch Slurm job to process one subject with freesurfer.

    Only keep cortical thickness stats.

    Args:
        sub_id (str): identifier of subject (with 'sub-')
        ses_id (str): identifier of session (with 'ses-')
        dataset (BIDSDirectoryDataset) : dataset object to retrieve pathes
        gen (str): generation identifiers

    Returns:
        int: sbatch launch code
    """
    logging.info("Processing subject : %s", sub_id)

    # Preparing Slurm Job parameters
    output_slurm = f"./logs/{sub_id}_{ses_id}"
    if gen_id is not None:
        output_slurm = f"{output_slurm}_{gen_id}"
    output_slurm = f"{output_slurm}.%j.out"
    job = Slurm(
        cpus_per_task=1,
        mem="30G",
        account="rrg-ebrahimi",
        time="20:00:00",
        output=output_slurm,
    )

    # Creating an Apptainer Command to use freesurfer with apptainer
    apptainer_env = ApptainerEnv.narval_freesurfer()
    job.add_cmd("module load apptainer")

    # Bind SLURM tmp dir to be used as freesurfer subject dir
    apptainer_env.bind("$SLURM_TMP_DIR", "/tmp")
    # Bind DATASET_ROOT to retrieve volumes and store final data
    apptainer_env.bind(config.DATASET_ROOT, "/data")

    # the directory where freesurfer will store results(needs to be unique)
    fs_dir = sub_id + ses_id

    # Create and add recon-all command to our apptainer command
    if gen_id is not None:
        fs_dir += gen_id
        freesurfer_cmd = FreeSurferReconAll(
            fs_dir,
            dataset.get_t1w(sub_id, ses_id, gen_id=gen_id),
            subject_dir="/tmp",
        )
    else:
        freesurfer_cmd = FreeSurferReconAll(
            fs_dir, dataset.get_t1w(sub_id, ses_id), subject_dir="/tmp"
        )
    apptainer_env.add_command(freesurfer_cmd.compile())
    job.add_cmd(apptainer_env.compile())

    # Create necessary directories
    derivative_dir = os.path.join(dataset.base_path, "derivatives")
    os.makedirs(derivative_dir, exist_ok=True)

    sub_dir = os.path.join(derivative_dir, sub_id, ses_id)
    if gen_id is not None:
        sub_dir = os.path.join(sub_dir, gen_id)

    os.makedirs(sub_dir, exist_ok=True)

    stat_dir = os.path.join(sub_dir, "stats")
    if os.path.exists(stat_dir):
        shutil.rmtree(stat_dir)
    os.makedirs(stat_dir)

    job.add_cmd(
        f"mv /tmp/{fs_dir}/stats/rh.aparc.stats \
              {os.path.join(stat_dir, 'rh.aparc.stats')}"
    )
    return job.sbatch(
        f"mv /tmp/{fs_dir}/stats/lh.aparc.stats \
              {os.path.join(stat_dir, 'lh.aparc.stats')}"
    )
