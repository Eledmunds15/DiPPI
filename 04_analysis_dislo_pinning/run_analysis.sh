#!/bin/bash
#SBATCH --job-name=Precipitate_Analysis
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --output=output-%j.log
#SBATCH --error=error-%j.log

# Load python modules
export SLURM_EXPORT_ENV=ALL
module load Anaconda3
source activate analysis

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run python script
mpirun -n $SLURM_NTASKS python analysis.py
