# --------------------------- LIBRARIES ---------------------------#
import os
import re
from mpi4py import MPI

from ovito.io import import_file, export_file
from ovito.modifiers import WignerSeitzAnalysisModifier, ExpressionSelectionModifier, DeleteSelectedModifier
from ovito.pipeline import FileSource

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

INPUT_DIR = '../05_jog_stability/dump_files'
INPUT_FILE = 'dumpfile_*'

OUTPUT_POINT_DEFECT_DIR = 'WS_point_defect_files'

REFERENCE_DIR = '../04_jog_creation/min_dump'
REFERENCE_FRAME = 'edge_dislo_1_dump'

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
        os.makedirs(OUTPUT_POINT_DEFECT_DIR, exist_ok=True)
        clear_dir(OUTPUT_POINT_DEFECT_DIR)

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
    output_paths = [os.path.join(OUTPUT_POINT_DEFECT_DIR, dump_file) for dump_file in dump_chunk]

    # Load the input files as a trajectory
    pipeline = import_file(input_paths)

    # Set up the Wigner-Seitz analysis
    ws_modifier = WignerSeitzAnalysisModifier()
    ws_modifier.reference = FileSource()
    ws_modifier.reference.load(os.path.join(REFERENCE_DIR, REFERENCE_FRAME))
    pipeline.modifiers.append(ws_modifier)

    # Set up the Expression Selection
    exp_modifier = ExpressionSelectionModifier(expression='Occupancy == 1')
    pipeline.modifiers.append(exp_modifier)

    # Set up the Delete Selection
    del_modifier = DeleteSelectedModifier()
    pipeline.modifiers.append(del_modifier)

    for frame in range(pipeline.num_frames):
        data = pipeline.compute(frame)

        # Export only the selected particles using the 'Selection' tag
        export_file(
            data,
            output_paths[frame],
            "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "c_peratom", "Occupancy"],
        )

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