#!/bin/bash
#SBATCH --job-name=Precipitate_Simulation
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --output=999_HPC_outputs/output-%j.log
#SBATCH --error=999_HPC_outputs/error-%j.log

# Load python modules
export SLURM_EXPORT_ENV=ALL
module load Anaconda3
source activate mol_dynamics_lmp

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run python script
mpirun -np $SLURM_NTASKS python -m 03_dislo_pin.simulate