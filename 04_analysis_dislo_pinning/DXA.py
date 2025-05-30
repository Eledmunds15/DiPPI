# --------------------------- LIBRARIES ---------------------------#
import os
import re
from mpi4py import MPI

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

INPUT_DIR = '../05_jog_stability/dump_files'
INPUT_FILE = 'dumpfile_*'

OUTPUT_LINES_DIR = 'DXA_lines_files'
OUTPUT_ATOMS_DIR = 'DXA_atoms_files'


AVERAGE_WINDOW = 5

# --------------------------- ANALYSIS ---------------------------#

def main():
    #--- INITIALISE MPI ---#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    set_path()

    #--- INITIALISE VARIABLE ON ALL RANKS ---#
    dump_files = None

    #--- CREATE AND SET DIRECTORIES ---#
    if rank == 0:
        os.makedirs(OUTPUT_LINES_DIR, exist_ok=True)
        os.makedirs(OUTPUT_ATOMS_DIR, exist_ok=True)
        clear_dir(OUTPUT_LINES_DIR)
        clear_dir(OUTPUT_ATOMS_DIR)

        dump_files = get_filenames(INPUT_DIR)

        print(f"Found {len(dump_files)} dump files to process.")
        print(f"Using {size} ranks for parallel processing.\n")

    #--- BROADCAST AND DISTRIBUTE WORK ---#
    dump_files = comm.bcast(dump_files, root=0)

    # Each rank gets only its share of files to process
    start, end = split_indexes(len(dump_files), rank, size)

    print(f"Rank {rank} of size {size} processing files from {start} to {end}")

    #--- PROCESS FILES ---#
    process_file(dump_files[start:end])
                
    return None

# --------------------------- UTILITIES ---------------------------#

def process_file(dump_chunk):

    input_paths = [os.path.join(INPUT_DIR, dump_file) for dump_file in dump_chunk]
    output_atoms_path = [os.path.join(OUTPUT_ATOMS_DIR, dump_file) for dump_file in dump_chunk]
    output_lines_path = [os.path.join(OUTPUT_LINES_DIR, dump_file) for dump_file in dump_chunk]

    pipeline = import_file(input_paths)

    DXA_modifier = DislocationAnalysisModifier()
    DXA_modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC

    # Add the time-averaging modifier:
    pipeline.modifiers.append(DXA_modifier)

    for frame in range(pipeline.num_frames):
        data = pipeline.compute(frame)

        export_file(pipeline, output_lines_path[frame], "ca")
        
        export_file(data, output_atoms_path[frame], "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "c_peratom", "c_csym", "Cluster"])
        
        print(f"Successfully processed frame {frame}...")

def view_information(data):
    
    print('')
    print("Available particle properties:")
    for prop in data.particles.keys():
        print(f"  - {prop}")

    print("\nAvailable global attributes:")
    for attr in data.attributes.keys():
        print(f"  - {attr}")
    print('')

def get_filenames(dir_path):
    """Returns a naturally sorted list of filenames (not paths) in the given directory."""
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return sorted(files, key=natural_sort_key)

def natural_sort_key(s):
    # Split the string into digit and non-digit parts, convert digits to int
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def split_indexes(n_files, rank, size):
    """Split n_files into contiguous chunks of indexes for each rank."""
    chunk_size = n_files // size
    remainder = n_files % size

    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    return [start, end]

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()