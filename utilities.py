import os
import sys

def set_path():
    """
    Set the current working directory to the directory of the running script.
    
    This is useful when you want to ensure that all relative file operations
    happen with respect to the script's location, not the current shell directory.
    """
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    # Uncomment below to debug the working directory being set
    # print(f"[DEBUG] Working directory set to: {script_dir}")

def clear_dir_exclude_file(exclude_file):
    current_script = os.path.basename(__file__)
    for filename in os.listdir('.'):
        if filename not in [current_script, exclude_file] and os.path.isfile(filename):
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Could not delete {filename}: {e}")