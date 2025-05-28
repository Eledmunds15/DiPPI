import os
import shutil
import numpy as np
from mpi4py import MPI

from ovito.io import import_file, export_file
from ovito.modifiers import DeleteSelectedModifier, InvertSelectionModifier

# --------------------------- CONFIG ---------------------------
DUMP_DIR = 'dump_files'
OUTPUT_DIR = 'dump_files_processed'
PRECIPITATE_ID_FILE = 'precipitate_ID'
PERATOM_THRESHOLD = -4.0
# -------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Preparing output directory...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clear_dir(OUTPUT_DIR)
        print(f"Loading precipitate IDs...")
        precipitate_ids = load_precipitate_ids(PRECIPITATE_ID_FILE)
        print(f"Loaded {len(precipitate_ids)} precipitate IDs.")
        
        dump_files = sorted([
            f for f in os.listdir(DUMP_DIR)
            if os.path.isfile(os.path.join(DUMP_DIR, f))
        ])
        print(f"Found {len(dump_files)} dump files to process.")
        print(f"Using {size} ranks for parallel processing.\n")
    else:
        precipitate_ids = None
        dump_files = None

    # Broadcast data
    precipitate_ids = comm.bcast(precipitate_ids, root=0)
    dump_files = comm.bcast(dump_files, root=0)

    # Each rank handles its portion of the files
    for i, dump_file in enumerate(dump_files):
        if i % size == rank:
            print(f"Rank {rank}: Processing file {dump_file} ({i+1}/{len(dump_files)})")
            process_dump_file(dump_file, precipitate_ids)
            print(f"Rank {rank}: Finished {dump_file}")

    comm.Barrier()
    if rank == 0:
        print("\nAll files processed successfully.")

# -------------------- Processing Functions --------------------

def process_dump_file(dump_file, precipitate_ids):
    input_path = os.path.join(DUMP_DIR, dump_file)
    output_path = os.path.join(OUTPUT_DIR, dump_file)

    pipeline = import_file(input_path)

    pipeline.modifiers.append(select_atoms)
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())

    for frame in range(pipeline.num_frames):
        data = pipeline.compute(frame)

        export_file(data, output_path, "lammps/dump", 
                    columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "c_peratom"])

# -------------------- Selection Function --------------------

def select_atoms(frame, data, precipitate_IDs=None):
    
    if precipitate_IDs is None:
        precipitate_IDs = load_precipitate_ids(PRECIPITATE_ID_FILE)
    
    ids = data.particles_['Particle Identifier']
    peratom = data.particles_['c_peratom']
    
    select_precipitate = (np.isin(ids, list(precipitate_IDs))) 
    select_high_energy_atoms = (peratom > PERATOM_THRESHOLD)

    selection = select_precipitate | select_high_energy_atoms

    data.particles.create_property('Selection', data=selection)

# ------------------------ Utilities ---------------------------

def clear_dir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def load_precipitate_ids(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    ids = set()
    reading_atoms = False

    for line in lines:
        if line.strip().startswith("ITEM: ATOMS"):
            reading_atoms = True
            continue
        elif line.strip().startswith("ITEM:"):
            reading_atoms = False
        elif reading_atoms:
            ids.add(int(line.strip()))

    return ids

# ------------------------ Entrypoint --------------------------

if __name__ == '__main__':
    main()
