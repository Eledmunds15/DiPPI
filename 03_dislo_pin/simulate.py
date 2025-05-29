# --------------------------- LIBRARIES ---------------------------#
import os
from mpi4py import MPI
import numpy as np
from lammps import lammps, PyLammps

from utilities import set_path, clear_dir

# --------------------------- CONFIG ---------------------------#

INPUT_DIR = '../02_minimization/min_input'
INPUT_FILE = 'edge_dislo.lmp'

DUMP_DIR = 'dump_files'
RESTART_DIR = 'restart_files'

POTENTIAL_DIR = '../00_potentials'
POTENTIAL_FILE = 'malerba.fs'

PRECIPITATE_RADIUS = 30
DISLOCATION_INITIAL_DISPLACEMENT = 10 # Distance from the precipitate in Angstroms
FIXED_SURFACE_DEPTH = 10 # Depth of the fixed surface in Angstroms

DT = 0.001
TEMPERATURE = 100
SHEAR_VELOCITY = 1

RUN_TIME = 100
DUMP_FREQ = 1000
RESTART_FREQ = 1000

# --------------------------- MINIMIZATION ---------------------------#

def main():

    #--- INITIALISE MPI ---#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    set_path()

    if rank == 0:
          os.makedirs(DUMP_DIR, exist_ok=True)
          clear_dir(DUMP_DIR)

    #--- CREATE AND SET DIRECTORIES ---#

    input_filepath = os.path.join(INPUT_DIR, INPUT_FILE)

    dump_file = 'dumpfile_*'
    restart_file = 'restart_*'

    dump_filepath = os.path.join(DUMP_DIR, dump_file)
    restart_filepath = os.path.join(RESTART_DIR, restart_file)

    potential_path = os.path.join(POTENTIAL_DIR, POTENTIAL_FILE)

    #--- LAMMPS Script ---#
    #--- Settings ---#
    lmp = lammps()
    L = PyLammps(ptr=lmp)

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
    L.velocity('top_surface', 'set', -SHEAR_VELOCITY, 0.0, 0.0)
    L.velocity('bottom_surface', 'set', SHEAR_VELOCITY, 0.0, 0.0)

    L.fix('precipitate_freeze', 'precipitate', 'setforce', 0.0, 0.0, 0.0)
    L.velocity('precipitate', 'set', 0.0, 0.0, 0.0)

    #--- Dump ID's for post-processing ---#
    L.write_dump('precipitate', 'custom', 'precipitate_ID', 'id')

    #--- Thermo ---#
    L.thermo_style('custom', 'step', 'temp', 'pe', 'etotal', 'c_press_comp[1]', 'c_press_comp[2]', 'c_press_comp[3]', 'c_press_comp[4]', 'c_press_comp[5]', 'c_press_comp[6]')
    L.thermo(1000)

    #--- Dump Files ---#
    L.dump('1', 'all', 'custom', DUMP_FREQ, dump_filepath, 'id', 'x', 'y', 'z', 'c_peratom', 'c_stress[4]')

    #--- Restart Files ---#
    L.restart(RESTART_FREQ, )

    L.run(RUN_TIME, restart_filepath)

    L.close()

    return None

# --------------------------- UTILITIES ---------------------------#

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()