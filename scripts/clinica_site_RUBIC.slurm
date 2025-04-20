#!/bin/bash
#SBATCH --job-name=RUBIC-prepoc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=249G
#SBATCH --time=24:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mail-user=bricout.charles@outlook.com
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output-%x.%j.out # STDOUT

# Load Python module (assuming it's installed on your cluster)
module load StdEnv/2020 gcc/9.3.0
module load ants/2.4.4
module load python/3.10.2

source ~/fix_clinica/bin/activate

clinica run t1-linear  ~/scratch/HBN_RUBIC ~/scratch/HBN_RUBIC_preproc -wd $SLURM_TMPDIR/HBN_RUBIC_workdir -np 25


