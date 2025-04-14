#!/bin/bash
#SBATCH --job-name=OpenNeuro-prepoc
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
mkdir ~/scratch/OpenNeuro_preproc

for dir in ~/scratch/OpenNeuro/ds*; do
    if [[ -d "$dir" ]]; then
        echo "Now Processing $dir"
        base=$(basename "$dir")
        clinica run t1-linear  $dir ~/scratch/OpenNeuro_preproc/$base -wd $SLURM_TMPDIR/$base_workdir -np 25 --uncropped_image
    fi
done

