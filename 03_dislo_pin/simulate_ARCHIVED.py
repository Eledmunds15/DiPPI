## Import libraries
import os
import shutil
import numpy as np
from mpi4py import MPI
from lammps import lammps, PyLammps

def main():

    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Identify Output Files#
    input_folder = 'minimized_files'
    output_folder = 'displaced_files'
    dump_folder = 'dump_files'

    if rank == 0:
        os.makedirs(dump_folder, exist_ok=True)
        clear_dir("dump_files")

    input_file = 'dislocation_precipitate_input_large.lmp'
    output_file = input_file
    dump_file = 'dumpFile_*'

    input_path = os.path.join(input_folder, input_file)
    output_path = os.path.join(output_folder, output_file)
    dump_path = os.path.join(dump_folder, dump_file)

    # Input Parameters

    precipitate_radius = 20
    dislocation_displacement = precipitate_radius+10
    fixed_surface_depth = 10

    dt = 0.001
    temperature = 600 # K
    shear_stress = 1e4 # Pa (J/m^3)

    #--- LAMMPS Script ---#
    #--- Settings ---#
    lmp = lammps()
    L = PyLammps(ptr=lmp)

    L.units('metal')
    L.atom_style('atomic')

    L.command('boundary p f p')

    L.read_data(input_path)

    L.pair_style('eam/fs')
    L.pair_coeff('*', '*', 'malerba.fs', 'Fe')

    #--- Get box bounds of the simulation ---#
    box_bounds = lmp.extract_box()

    box_min = box_bounds[0]
    box_max = box_bounds[1]

    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]

    sim_box_center = [np.mean([xmin, xmax]), np.mean([ymin, ymax]), np.mean([zmin, zmax])]

    #--- Calculate Force in LAMMPS-relevant units ---#

    J_to_eV = 6.241509e18         # 1 J = 6.241509e18 eV
    m3_to_ang3 = 1e30             # 1 m^3 = 1e30 Angstrom^3

    ev_per_ang3 = shear_stress * J_to_eV / m3_to_ang3 # Energy to put into the system in ev/Angstrom^3

    area = (zmax-zmin)*(xmax-xmin) # In Angstroms

    force_atomic = ev_per_ang3*area # Calculate the force to be applied (total) by multiplying by atomic area (along x and z-axis)
    force_atomic_surf = force_atomic/2

    #--- Displace Atoms ---#

    L.group('all', 'type', 1)

    L.displace_atoms('all', 'move', dislocation_displacement, 0, 0, 'units', 'box')

    #--- Defining Regions ---#
    L.region('precipitate_reg', 'sphere', sim_box_center[0], sim_box_center[1], sim_box_center[2], precipitate_radius)
    L.region('top_surface_reg', 'block', 'INF', 'INF', (ymax-fixed_surface_depth), 'INF', 'INF', 'INF')
    L.region('bottom_surface_reg', 'block', 'INF', 'INF', 'INF', (ymin+fixed_surface_depth), 'INF', 'INF')

    #--- Define Groups ---#
    L.group('top_surface', 'region', 'top_surface_reg')
    L.group('bottom_surface', 'region', 'bottom_surface_reg')
    L.group('precipitate', 'region', 'precipitate_reg')
    L.group('mobile_atoms', 'subtract', 'all', 'precipitate', 'top_surface', 'bottom_surface')

    #--- Define Number of Atoms ---#
    L.variable("n_top", "equal", "count(top_surface)")
    L.variable("n_bottom", "equal", "count(bottom_surface)")
    L.variable("n_obstacle", "equal", "count(precipitate)")

    n_top = lmp.extract_variable('n_top')
    n_bottom = lmp.extract_variable('n_bottom')
    n_obstacle = lmp.extract_variable('n_obstacle')

    #--- Define Forces for Fixed Surfaces ---#
    force_top = force_atomic_surf/n_top
    force_bottom = force_atomic_surf/n_bottom

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
    L.fix('1', 'all', 'nvt', 'temp', temperature, temperature, 100.0*dt)
    
    L.velocity('mobile_atoms', 'create', temperature, 1234, 'mom', 'yes', 'rot', 'yes')

    # Define fixes and forces for the top and bottom surfaces
    L.fix('top_surface_freeze', 'top_surface', 'setforce', -force_top, 0.0, 0.0)
    L.fix('bottom_surface_freeze', 'bottom_surface', 'setforce', force_bottom, 0.0, 0.0)
    L.velocity('top_surface', 'set', 0.0, 0.0, 0.0)
    L.velocity('bottom_surface', 'set', 0.0, 0.0, 0.0)

    L.fix('precipitate_freeze', 'precipitate', 'setforce', 0.0, 0.0, 0.0)
    L.velocity('precipitate', 'set', 0.0, 0.0, 0.0)

    #--- Dump ID's for post-processing ---#
    L.write_dump('precipitate', 'custom', 'precipitate_ID', 'id')

    #--- Thermo ---#
    L.thermo_style('custom', 'step', 'temp', 'pe', 'etotal', 'c_press_comp[1]', 'c_press_comp[2]', 'c_press_comp[3]', 'c_press_comp[4]', 'c_press_comp[5]', 'c_press_comp[6]')
    L.thermo(200)

    #--- Dump Files ---#
    L.dump('1', 'all', 'custom', 1000, dump_path, 'id', 'x', 'y', 'z', 'c_peratom', 'c_stress[4]')

    L.run(1000000)

    L.close()

def clear_dir(dir_path):

    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a valid directory")

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":

    main()
