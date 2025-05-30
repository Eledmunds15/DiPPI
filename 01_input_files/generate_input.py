# --------------------------- LIBRARIES ---------------------------#
import os
import shutil
import subprocess as sp
from matscipy.calculators.eam import EAM
from matscipy.dislocation import get_elastic_constants

from utilities import set_path, clear_dir_exclude_file

# --------------------------- CONFIG ---------------------------#

POTENTIAL_DIR = '../00_potentials'
POTENTIAL_FILE = 'malerba.fs'

OUTPUT_FILE = 'edge_dislo.lmp'

MATERIAL = 'Fe'
CRYSTAL_STRUCTURE = 'bcc'
POIS_RATIO = 0.3

X_ORIENTATION = '[111]'
Y_ORITENTATION = '[-101]'
Z_ORITENTATION = '[1-21]'

X_SIZE = 100
Y_SIZE = 70
Z_SIZE = 60

# --------------------------- INPUT GENERATOR ---------------------------#
def main():

    set_path()

    clear_dir_exclude_file()

    #--- DEFINE DIRECTORIES ---#

    potential_path = os.path.join(POTENTIAL_DIR, POTENTIAL_FILE)

    #--- PREPARE INPUTS ---#

    eam_calc = EAM(potential_path)
    lat_const, _, _, _ = get_elastic_constants(calculator=eam_calc, symbol=MATERIAL, verbose='False')
    LATTICE_CONSTANT = str(lat_const)

    A_y = str(int(Y_SIZE/2))
    B_y = str(int(Y_SIZE/2))

    def_ten = str(float(0.5/X_SIZE))
    def_comp = str(float(0.5/(X_SIZE+1)))

    A_x = str(X_SIZE+1)
    B_x = str(X_SIZE)

    A_z = str(Z_SIZE)
    B_z = str(Z_SIZE)

    #--- DEFINE SUBPROCESS COMMANDS ---#

    create_unitlattice = ['atomsk', '--create', CRYSTAL_STRUCTURE, LATTICE_CONSTANT, MATERIAL, 'orient', X_ORIENTATION, Y_ORITENTATION, Z_ORITENTATION, 'unitcell.xsf']

    create_superlattice_A = ['atomsk', 'unitcell.xsf', '-duplicate', A_x, A_y, A_z, '-deform', 'X', def_ten, '0.0', 'supercell_A.xsf']
    create_superlattice_B = ['atomsk', 'unitcell.xsf', '-duplicate', B_x, B_y, B_z, '-deform', 'X', def_comp, '0.0', 'supercell_B.xsf']

    merge_col = ['atomsk', '--merge', 'Y', '2', 'supercell_A.xsf', 'supercell_B.xsf', 'dislocation_col.xsf']

    wrap_box = ['atomsk', 'dislocation_col.xsf', '-wrap', OUTPUT_FILE]

    #--- EXECUTE COMMANDS ---#
    sp.run(create_unitlattice)
    sp.run(create_superlattice_A)
    sp.run(create_superlattice_B)
    sp.run(merge_col)
    sp.run(wrap_box)

    clear_dir_exclude_file(OUTPUT_FILE)

    return None

# --------------------------- UTILITIES ---------------------------#

# --------------------------- ENTRY POINT ---------------------------#
if __name__ == '__main__':

    main()