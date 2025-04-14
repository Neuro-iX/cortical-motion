import sys
from functools import wraps
from typing import Iterable, Sequence

import click
from simple_slurm import Slurm

from src import config
from src.utils.click import get_command

slurm_arg = click.option(
    "--slurm", "-S", help="Launch fonction as Slurm job", is_flag=True
)


def slurm_adaptor(
    n_cpus: int = 1,
    n_gpus: int = 0,
    mem: str = "8G",
    time: str = "1:00:00",
    cpy_synth_ds=False,
):
    """This function generate a decorate to enable slurm on a click command.
    it retrieves the command used to launch the program and launch it in a python slurm job
    Use the --slurm / -S flag to trigger slurm.

    Args:
        n_cpus (int, optional): Number of cpus. Defaults to 1.
        n_gpus (int, optional): Number of gpus. Defaults to 0.
        mem (str, optional): Memory to allocate. Defaults to "8G".
        time (str, optional): Time to reserve. Defaults to "1:00:00".
    """

    def decorator(func):
        @slurm_arg
        @wraps(func)
        def wrapper(slurm: bool, *args, **kwargs):
            if slurm:
                job = get_python_slurm(
                    func.__name__,
                    None,
                    output=f"./logs/{func.__name__}.%j.out",
                    n_cpus=n_cpus,
                    n_gpus=n_gpus,
                    mem=mem,
                    time=time,
                )
                if cpy_synth_ds:
                    print("add tmp data command")
                    copy_data_tmp(job, ["SynthCortical"])
                launch_as_slurm(job)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def slurm_loop_adaptor(
    iterator: Iterable[any],
    param: str,
    n_cpus: int = 1,
    n_gpus: int = 0,
    mem: str = "8G",
    time: str = "1:00:00",
):
    def decorator(func):
        @slurm_arg
        @click.pass_context
        @wraps(func)
        def wrapper(ctx: click.Context, slurm: bool, *args, **kwargs):
            if slurm:
                for element in iterator:
                    job = get_python_slurm(
                        f"{func.__name__}-{element}",
                        None,
                        output=f"./logs/{func.__name__}-{element}.%j.out",
                        n_cpus=n_cpus,
                        n_gpus=n_gpus,
                        mem=mem,
                        time=time,
                    )
                    print(wrapper)
                    cmd = get_command(ctx, func, **{param: element})
                    launch_as_slurm(job, f"python {cmd}")
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def launch_as_slurm(slurm_job: Slurm, full_command: str = None):
    """Launch the command used to execute script as a slurm job

    Args:
        slurm_job (Slurm): Slurm job to launch command on
        full_command (str, optional): full command may be provide or will be retrieved. Defaults to None.
    """
    if full_command is None:
        full_command = " ".join(sys.argv)
        full_command = f"srun python3 {full_command}"

    full_command = full_command.replace("-S", "").replace("--slurm", "")
    print(slurm_job)
    print(full_command)
    slurm_job.sbatch(full_command)


def get_python_slurm(
    name: str,
    array: Sequence[int] | int | None,
    output: str,
    n_cpus: int,
    n_gpus: int,
    account=config.DEFAULT_SLURM_ACCOUNT,
    mem="300G",
    time="24:00:00",
    nodes=1,
) -> Slurm:
    """Generate a basic job with requeu and python setup

    Args:
        name (str): Job name
        array (Sequence[int] | int | None): Array parameter: single id, sequence, range or nothing
        output (str): Output path
        n_cpus (int): Number of CPUs to allocate
        n_gpus (int): Number of GPUs to allocate
        account (_type_, optional): Account id to use. Defaults to DEFAULT_SLURM_ACCOUNT
            (see config.py).
        mem (str, optional): RAM to allocate. Defaults to "200G".
        time (str, optional): Time to allocate ressources for. Defaults to "24:00:00".

    Returns:
        Slurm: The job with requeu enabled and a ready python environment
    """
    job = Slurm(
        job_name=name,
        nodes=nodes,
        cpus_per_task=n_cpus,
        ntasks_per_node=max(n_gpus, 1),
        ntasks=max(n_gpus, 1),
        mem=mem,
        time=time,
        account=account,
        signal="SIGUSR1@90",
        requeue=True,
        output=output,
    )
    if n_gpus > 0:
        job.add_arguments(gpus_per_node=n_gpus)
    if not array is None:
        job.add_arguments(array=array)
    setup_python(job)
    return job


def setup_python(job: Slurm):
    """Add command to a slurm job to activate necessary modules and environments


    Args:
        job (Slurm): slurm job to modify
    """
    job.add_cmd("module load python cuda httpproxy opencv")
    job.add_cmd("source ~/fix_bowl/bin/activate")
    job.add_cmd('echo "python is setup"')


def copy_data_tmp(job: Slurm, tar_files: list[str]):
    """Extract data from scratch to $SLURM_TMPDIR for pretraining dataset

    Args:
        job (Slurm): slurm job to modify
    """
    job.add_cmd("export DATASET_ROOT=$SLURM_TMPDIR/datasets")
    job.add_cmd("mkdir -p $SLURM_TMPDIR/datasets")
    for ds in tar_files:
        job.add_cmd(
            f"tar --skip-old-file -xf /home/cbricout/projects/ctb-sbouix/cbricout/cortical-motion-datasets/{ds}.tar -C $SLURM_TMPDIR/datasets "
        )
        job.add_cmd(f'echo "{ds} copied"')

        job.add_cmd(f"head $SLURM_TMPDIR/datasets/SynthCortical/scores.csv")
