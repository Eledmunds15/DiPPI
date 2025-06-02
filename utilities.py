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

def clear_dir_exclude_files(target_dir, exclude_files=None):
    exclude_files = set(os.path.abspath(f) for f in (exclude_files or []))
    
    for filename in os.listdir(target_dir):
        filepath = os.path.abspath(os.path.join(target_dir, filename))
        
        if filepath in exclude_files:
            continue
        
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted: {filepath}")
            except Exception as e:
                print(f"Could not delete {filepath}: {e}")

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