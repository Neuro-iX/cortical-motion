#!/bin/bash
#SBATCH --job-name=HCPEP_preproc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=249G
#SBATCH --time=3:00:00
#SBATCH --account=ctb-sbouix
#SBATCH -o ./logs/output-%x.%j.out # STDOUT

# Load Python module (assuming it's installed on your cluster)
module load StdEnv/2020 gcc/9.3.0
module load ants/2.4.4
module load python/3.10.2

source ~/fix_clinica/bin/activate

clinica run t1-linear  ~/scratch/HCPEP ~/scratch/HCPEP_preproc  -wd $SLURM_TMPDIR/HCPEP_workdir -np 25 --uncropped_image


