#!/bin/bash
#SBATCH --job-name=Precipitate_Study
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=1500MB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=512
#SBATCH --output=output-%j.log
#SBATCH --error=error-%j.log

# Print some info
echo "SLURM_NTASKS = $SLURM_NTASKS"

# Load python modules
export SLURM_EXPORT_ENV=ALL
module load Anaconda3
source activate lammps

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run python script
mpirun -n $SLURM_NTASKS python simulate.py
