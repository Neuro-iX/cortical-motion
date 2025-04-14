#!/bin/bash
#SBATCH --job-name=compress-synth
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --account=ctb-sbouix
#SBATCH --mail-user=bricout.charles@outlook.com
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output-%x.%j.out # STDOUT

cd $SLURM_TMPDIR
tar -cf  SynthCortical_wider.tar -P /scratch/cbricout/SynthCortical_wider/


mv $SLURM_TMPDIR/SynthCortical_wider.tar /scratch/cbricout/