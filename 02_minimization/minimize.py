# --------------------------- LIBRARIES ---------------------------#
import os
from mpi4py import MPI
from lammps import lammps, PyLammps

# --------------------------- CONFIG ---------------------------#

INPUT_DIR = '../01_input_files'
INPUT_FILE = 'straight_edge_dislo.lmp'

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

    #--- CREATE AND SET DIRECTORIES ---#

    input_filepath = os.path.join(INPUT_DIR, INPUT_FILE)

    output_file = 'straight_edge_dislo.lmp'
    dump_file = 'straight_edge_dislo_dump'

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

def set_path():

    filepath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    # print(f'Working directory set to: {filepath}')

def clear_dir(dir_path):
     
    if not os.path.isdir(dir_path):
        raise ValueError(f"The path '{dir_path}' is not a directory or does not exist.")
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                # Optional: if you want to delete subdirectories too
                for root, dirs, files in os.walk(file_path, topdown=False):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def initialize_directories():

    set_path() # Set path to location of current directory

    os.makedirs(DUMP_DIR, exist_ok=True) # Create the dump directory
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the output directory

    clear_dir(DUMP_DIR) # Clear dump directory of previous data
    clear_dir(OUTPUT_DIR) # Clear output_director of previous data

    print("Directories successfully initialized...")

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        initialize_directories()
        main()