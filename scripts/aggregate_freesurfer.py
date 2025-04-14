import os
import sys

sys.path.append(".")
from simple_slurm import Slurm

from src.utils.slurm import launch_as_slurm

datasets = ["HCPEP_preproc", "MRART_preproc"]


def process_ds(ds, time="1:00:00"):
    job = Slurm(
        job_name=f"{ds}_agg_all_fs",
        nodes=1,
        cpus_per_task=1,
        mem="20G",
        time=time,
        account="rrg-ebrahimi",
        output="./logs/output-%x.%j.out",
    )
    job.add_cmd("module load apptainer")
    job.sbatch(
        f"""
    cd /home/cbricout/scratch/{ds}/subjects/derivatives
    printf "%s\n" $(dirname $(dirname sub-*/ses-*/stats/lh.aparc*)) > subjects.txt
    apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/lh.area.tsv --hemi lh"
    apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/rh.area.tsv --hemi rh"
    apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/lh.thickness.tsv --hemi lh --meas thickness"
    apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/rh.thickness.tsv --hemi rh --meas thickness"
    rm subjects.txt
    """
    )


# def process_ds(ds, time="2:00:00"):
#     job = Slurm(
#         job_name=f"{ds}_agg_all_fs",
#         nodes=1,
#         cpus_per_task=1,
#         mem="20G",
#         time=time,
#         account="ctb-sbouix",
#         output="./logs/output-%x.%j.out",
#     )
#     job.add_cmd("module load apptainer")
#     job.sbatch(
#         f"""
#     cd /home/cbricout/scratch/{ds}/subjects/derivatives
#     printf "%s\n" $(dirname $(dirname sub-*/ses-*/stats/lh.aparc*)) > subjects.txt
#     apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/lh.gauscurv.tsv --hemi lh --meas gauscurv"
#     apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/rh.gauscurv.tsv --hemi rh --meas gauscurv"
#     apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/lh.volume.tsv --hemi lh --meas volume"
#     apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/rh.volume.tsv --hemi rh --meas volume"
#     apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/lh.meancurv.tsv --hemi lh --meas meancurv"
#     apptainer run --env "SUBJECTS_DIR=/subdir" --bind /home/cbricout/scratch/{ds}/subjects/derivatives:/subdir --bind /home/cbricout/scratch/{ds}:/out ~/projects/def-sbouix/software/Freesurfer7.4.1_container/fs741.sif bash -c "aparcstats2table --subjectsfile=subjects.txt  -t /out/rh.meancurv.tsv --hemi rh --meas meancurv"

#     rm subjects.txt
#     """
#     )


for ds in datasets:
    process_ds(ds)

# for ds in os.listdir("/home/cbricout/scratch/ProcessedOpenNeuro"):
#    process_ds(f"ProcessedOpenNeuro/{ds}", time="1:00:00")
