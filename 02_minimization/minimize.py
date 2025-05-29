# --------------------------- LIBRARIES ---------------------------#
import os
from mpi4py import MPI
from lammps import lammps, PyLammps

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

INPUT_DIR = '../01_input_files'
INPUT_FILE = 'edge_dislo.lmp'

DUMP_DIR = 'min_dump'
OUTPUT_DIR = 'min_input'

POTENTIAL_DIR = '../00_potentials'
POTENTIAL_FILE = 'malerba.fs'

ENERGY_TOL = 1e-7
FORCE_TOL = 1e-10

# --------------------------- MINIMIZATION ---------------------------#

def main():

    #--- INITIALISE MPI ---#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    set_path()

    if rank == 0:
          os.makedirs(DUMP_DIR, exist_ok=True)
          os.makedirs(OUTPUT_DIR, exist_ok=True)
          clear_dir(DUMP_DIR)
          clear_dir(OUTPUT_DIR)

    #--- CREATE AND SET DIRECTORIES ---#

    input_filepath = os.path.join(INPUT_DIR, INPUT_FILE)

    output_file = 'edge_dislo.lmp'
    dump_file = 'edge_dislo_dump'

    output_filepath = os.path.join(OUTPUT_DIR, output_file)
    dump_filepath = os.path.join(DUMP_DIR, dump_file)

    potential_path = os.path.join(POTENTIAL_DIR, POTENTIAL_FILE)

    #--- LAMMPS SCRIPT ---#
    lmp = lammps()
    L = PyLammps(ptr=lmp)

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