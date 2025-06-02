# --------------------------- LIBRARIES ---------------------------#
import os
from mpi4py import MPI
from lammps import lammps, PyLammps

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

MASTER_DATA_DIR = '000_output_files'
MODULE_DIR = '02_minimization'

INPUT_DIR = '01_input'
INPUT_FILE = 'edge_dislo.lmp'

DUMP_DIR = 'min_dump'
OUTPUT_DIR = 'min_input'

POTENTIAL_DIR = '00_potentials'
POTENTIAL_FILE = 'malerba.fs'

ENERGY_TOL = 1e-6
FORCE_TOL = 1e-8

# --------------------------- MINIMIZATION ---------------------------#

def main():

    #--- INITIALISE MPI ---#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    set_path(PROJECT_ROOT)

    if rank == 0:
        os.makedirs(MASTER_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(MASTER_DATA_DIR, MODULE_DIR), exist_ok=True)

        dump_dir = os.path.join(MASTER_DATA_DIR, MODULE_DIR, DUMP_DIR)
        output_dir = os.path.join(MASTER_DATA_DIR, MODULE_DIR, OUTPUT_DIR)

        os.makedirs(dump_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        clear_dir(dump_dir)
        clear_dir(output_dir)

        input_filepath = os.path.join(MASTER_DATA_DIR, INPUT_DIR, INPUT_FILE)

        output_file = 'edge_dislo.lmp'
        dump_file = 'edge_dislo_dump'

        output_filepath = os.path.join(output_dir, output_file)
        dump_filepath = os.path.join(dump_dir, dump_file)

        potential_path = os.path.join(POTENTIAL_DIR, POTENTIAL_FILE)

    else:
        # For other ranks, initialize variables to None or empty strings
        dump_dir = None
        output_dir = None
        input_filepath = None
        output_filepath = None
        dump_filepath = None
        potential_path = None

    # Now broadcast all variables from rank 0 to all ranks
    dump_dir = comm.bcast(dump_dir, root=0)
    output_dir = comm.bcast(output_dir, root=0)
    input_filepath = comm.bcast(input_filepath, root=0)
    output_filepath = comm.bcast(output_filepath, root=0)
    dump_filepath = comm.bcast(dump_filepath, root=0)
    potential_path = comm.bcast(potential_path, root=0)

    #--- LAMMPS SCRIPT ---#
    lmp = lammps()
    L = PyLammps(ptr=lmp)

    L.log(os.path.join(MASTER_DATA_DIR, MODULE_DIR, 'log.lammps'))

    L.units('metal') # Set units style
    L.atom_style('atomic') # Set atom style

    L.command('boundary f f p') # Set the boundaries of the simulation

    L.read_data(input_filepath) # Read input file

    L.pair_style('eam/fs') # Set the potential style
    L.pair_coeff('*', '*', potential_path, 'Fe') # Select the potential

    L.group('fe_atoms', 'type', 1) # Group all atoms

    L.compute('peratom', 'all', 'pe/atom') # Set a compute to track the peratom energy

    L.minimize(ENERGY_TOL, FORCE_TOL, 1000, 10000) # Execute minimization

    L.write_dump('all', 'custom', dump_filepath, 'id', 'x', 'y', 'z', 'c_peratom') # Write a dumpfile containing atom positions and pot energies
    L.write_data(output_filepath) # Write a lammps input file with minimized configuration for subsequent sims

    L.close()

    return None

# --------------------------- UTILITIES ---------------------------#

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()