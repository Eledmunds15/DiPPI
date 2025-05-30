# --------------------------- LIBRARIES ---------------------------#
import os
import re
from mpi4py import MPI

from ovito.io import import_file, export_file
from ovito.modifiers import TimeAveragingModifier

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

INPUT_DIR = '../05_jog_stability/dump_files'
INPUT_FILE = 'dumpfile_*'

OUTPUT_DIR = 'time_averaged_files'

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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clear_dir(OUTPUT_DIR)

        dump_files = get_filenames(INPUT_DIR)

        print(f"Found {len(dump_files)} dump files to process.")
        print(f"Using {size} ranks for parallel processing.\n")

    #--- BROADCAST AND DISTRIBUTE WORK ---#
    dump_files = comm.bcast(dump_files, root=0)

    # Each rank gets only its share of files to process
    dump_index_rank = split_indexes(len(dump_files), rank, size)

    print(f"Rank {rank} of size {size} processing {dump_index_rank}")

    comm.Barrier()

    #--- PROCESS FILES ---#
    for index in dump_index_rank:

        dump_chunk = dump_files[index:index+AVERAGE_WINDOW]
        # print(dump_chunk)

        if len(dump_chunk) != AVERAGE_WINDOW:
            print("Ran out of files!")
            break

        process_files(dump_chunk)
        print(f"Successfully processe frame {index}...")
        
    return None

# --------------------------- UTILITIES ---------------------------#

def process_files(dump_chunk):

    input_paths = [os.path.join(INPUT_DIR, dump_file) for dump_file in dump_chunk]
    output_path = os.path.join(OUTPUT_DIR, dump_chunk[0])

    # Load the trajectory (make sure it's multiple frames!)
    pipeline = import_file(input_paths)

    # Add the time-averaging modifier:
    pipeline.modifiers.append(
        TimeAveragingModifier(
            operate_on = ('property:particles/c_csym', 'property:particles/c_peratom')
        )
    )

    # Evaluate at the last frame (you can change this)
    data = pipeline.compute(pipeline.source.num_frames - 1)

    export_file(data, output_path, "lammps/dump",
                columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "c_peratom", "c_peratom Average", "c_csym", "c_csym Average"])

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

    return list(range(start, end))

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()