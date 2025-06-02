# --------------------------- LIBRARIES ---------------------------#
import os
from mpi4py import MPI
import numpy as np
from lammps import lammps, PyLammps

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

MASTER_DATA_DIR = '000_output_files'
MODULE_DIR = '03_dislo_pin'

INPUT_DIR = '02_minimization/min_input'
INPUT_FILE = 'edge_dislo.lmp'

DUMP_DIR = 'dump_files'
RESTART_DIR = 'restart_files'

POTENTIAL_DIR = '00_potentials'
POTENTIAL_FILE = 'malerba.fs'

PRECIPITATE_RADIUS = 30
DISLOCATION_INITIAL_DISPLACEMENT = 10 # Distance from the precipitate in Angstroms
FIXED_SURFACE_DEPTH = 5 # Depth of the fixed surface in Angstroms

DT = 0.001
TEMPERATURE = 100
SHEAR_VELOCITY = 1

RUN_TIME = 100
THERMO_FREQ = 1000
DUMP_FREQ = 1000
RESTART_FREQ = 10000

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
        output_dir = os.path.join(MASTER_DATA_DIR, MODULE_DIR, RESTART_DIR)

        os.makedirs(dump_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        clear_dir(dump_dir)
        clear_dir(output_dir)

        input_filepath = os.path.join(MASTER_DATA_DIR, INPUT_DIR, INPUT_FILE)

        dump_file = 'dumpfile_*'
        restart_file = 'restart.*'

        restart_filepath = os.path.join(output_dir, restart_file)
        dump_filepath = os.path.join(dump_dir, dump_file)

        potential_path = os.path.join(POTENTIAL_DIR, POTENTIAL_FILE)

    else:
        # For other ranks, initialize variables to None or empty strings
        dump_dir = None
        output_dir = None
        input_filepath = None
        restart_filepath = None
        dump_filepath = None
        potential_path = None

    # Now broadcast all variables from rank 0 to all ranks
    dump_dir = comm.bcast(dump_dir, root=0)
    output_dir = comm.bcast(output_dir, root=0)
    input_filepath = comm.bcast(input_filepath, root=0)
    restart_filepath = comm.bcast(restart_filepath, root=0)
    dump_filepath = comm.bcast(dump_filepath, root=0)
    potential_path = comm.bcast(potential_path, root=0)

    #--- LAMMPS Script ---#
    #--- Settings ---#
    lmp = lammps()
    L = PyLammps(ptr=lmp)

    L.log(os.path.join(MASTER_DATA_DIR, MODULE_DIR, 'log.lammps'))

    L.units('metal')
    L.atom_style('atomic')

    L.command('boundary p f p')

    L.read_data(input_filepath)

    L.pair_style('eam/fs')
    L.pair_coeff('*', '*', potential_path, 'Fe')

    #--- Get box bounds of the simulation ---#
    box_bounds = lmp.extract_box()

    box_min = box_bounds[0]
    box_max = box_bounds[1]

    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]

    sim_box_center = [np.mean([xmin, xmax]), np.mean([ymin, ymax]), np.mean([zmin, zmax])]

    #--- Displace Atoms ---#

    L.group('all', 'type', 1)

    L.displace_atoms('all', 'move', PRECIPITATE_RADIUS+DISLOCATION_INITIAL_DISPLACEMENT, 0, 0, 'units', 'box')

    #--- Defining Regions ---#
    L.region('precipitate_reg', 'sphere', sim_box_center[0], sim_box_center[1], sim_box_center[2], PRECIPITATE_RADIUS)
    L.region('top_surface_reg', 'block', 'INF', 'INF', (ymax-FIXED_SURFACE_DEPTH), 'INF', 'INF', 'INF')
    L.region('bottom_surface_reg', 'block', 'INF', 'INF', 'INF', (ymin+FIXED_SURFACE_DEPTH), 'INF', 'INF')

    #--- Define Groups ---#
    L.group('top_surface', 'region', 'top_surface_reg')
    L.group('bottom_surface', 'region', 'bottom_surface_reg')
    L.group('precipitate', 'region', 'precipitate_reg')
    L.group('mobile_atoms', 'subtract', 'all', 'precipitate', 'top_surface', 'bottom_surface')

    #--- Define Computes ---#
    L.compute('peratom', 'all', 'pe/atom')
    L.compute('stress', 'all', 'stress/atom', 'NULL')
    L.compute('temp_compute', 'all', 'temp')
    L.compute('press_comp', 'all', 'pressure', 'temp_compute')

    L.compute('precipitate_force_x', 'precipitate', 'reduce', 'sum', 'fx')
    L.compute('precipitate_force_y', 'precipitate', 'reduce', 'sum', 'fy')
    L.compute('precipitate_force_z', 'precipitate', 'reduce', 'sum', 'fz')

    L.compute('precipitate_velocity_x', 'precipitate', 'reduce', 'sum', 'vx')
    L.compute('precipitate_velocity_y', 'precipitate', 'reduce', 'sum', 'vy')
    L.compute('precipitate_velocity_z', 'precipitate', 'reduce', 'sum', 'vz')

    #--- Define Fixes and Velocities ---#
    L.fix('1', 'all', 'nvt', 'temp', TEMPERATURE, TEMPERATURE, 100.0*DT)
    
    L.velocity('mobile_atoms', 'create', TEMPERATURE, 1234, 'mom', 'yes', 'rot', 'yes')

    # Define fixes and forces for the top and bottom surfaces
    L.fix('top_surface_freeze', 'top_surface', 'setforce', 0.0, 0.0, 0.0)
    L.fix('bottom_surface_freeze', 'bottom_surface', 'setforce', 0.0, 0.0, 0.0)
    L.velocity('top_surface', 'set', -(SHEAR_VELOCITY/2), 0.0, 0.0)
    L.velocity('bottom_surface', 'set', (SHEAR_VELOCITY/2), 0.0, 0.0)

    L.fix('precipitate_freeze', 'precipitate', 'setforce', 0.0, 0.0, 0.0)
    L.velocity('precipitate', 'set', 0.0, 0.0, 0.0)

    #--- Dump ID's for post-processing ---#
    L.write_dump('precipitate', 'custom', os.path.join(MASTER_DATA_DIR, MODULE_DIR, 'precipitate_ID'), 'id')

    #--- Thermo ---#
    L.thermo_style('custom', 'step', 'temp', 'pe', 'etotal', 'c_press_comp[1]', 'c_press_comp[2]', 'c_press_comp[3]', 'c_press_comp[4]', 'c_press_comp[5]', 'c_press_comp[6]')
    L.thermo(THERMO_FREQ)

    #--- Dump Files ---#
    L.dump('1', 'all', 'custom', DUMP_FREQ, dump_filepath, 'id', 'x', 'y', 'z', 'c_peratom', 'c_stress[4]')

    #--- Restart Files ---#
    L.restart(RESTART_FREQ, restart_filepath)

    L.run(RUN_TIME)

    L.close()

    return None

# --------------------------- UTILITIES ---------------------------#

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()