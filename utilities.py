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

def clear_dir_exclude_file(exclude_file=None):
    current_script = os.path.basename(sys.argv[0])
    print(current_script)
    for filename in os.listdir('.'):
        if filename == current_script:
            continue
        if exclude_file and filename == exclude_file:
            continue
        if os.path.isfile(filename):
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Could not delete {filename}: {e}")

def clear_dir(dir_path):
     
    if not os.path.isdir(dir_path):
        raise ValueError(f"The path '{dir_path}' is not a directory or does not exist.")
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                # Optional: if you want to delete subdirectories too
                for root, dirs, files in os.walk(file_path, topdown=False):
                    for f in files:
                        os.unlink(os.path.join(root, f))
                    for d in dirs:
                        os.rmdir(os.path.join(root, d))
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")