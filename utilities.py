import os

def set_path():

    filepath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(filepath)
    print(f'Working directory set to: {filepath}')

def clear_dir_exclude_file(exclude_file):
    current_script = os.path.basename(__file__)
    for filename in os.listdir('.'):
        if filename not in [current_script, exclude_file] and os.path.isfile(filename):
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Could not delete {filename}: {e}")