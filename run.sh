#!/bin/bash
#SBATCH --job-name=Precipitate_Analysis
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1500MB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --output=output-%j.log
#SBATCH --error=error-%j.log

# Create or clear HPC_outputs directory
if [ -d HPC_outputs ]; then
    rm -rf HPC_outputs/*
else
    mkdir HPC_outputs
fi

# Load python modules
export SLURM_EXPORT_ENV=ALL
module load Anaconda3
conda activate mol_dynamics_lmp

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run python script
mpirun -np $SLURM_NTASKS python -m 03_dislo_pin.simulate