import os
import shutil
import random
import pathlib

# Constants 
CWD = pathlib.Path(os.path.abspath(__file__))
GIT_ROOT = CWD.parent.parent.parent
DATA_DIR = GIT_ROOT / "data" / 'AIED2024'
REID_DB = DATA_DIR / 'reid' / 'db'

def copy_random_pngs(src_dir: pathlib.Path, dst_dir: pathlib.Path, N: int):
    # Ensure the source directory exists
    if not src_dir.exists():
        print(f"The source directory {src_dir} does not exist.")
        return

    # Ensure the destination directory exists, if not, create it
    os.makedirs(dst_dir, exist_ok=True)

    # Get all the .png files in the source directory
    png_files = [file for file in src_dir.iterdir() if file.suffix == '.png']

    # If there are fewer files than N, adjust N to the number of available files
    N = min(N, len(png_files))

    # Select N random .png files
    selected_files = random.sample(png_files, N)

    # Copy the selected files to the destination directory
    for file in selected_files:
        src_file_path = src_dir / file.name
        dst_file_path = dst_dir / file.name
        shutil.copy(src_file_path, dst_file_path)
        print(f"Copied {file} to {dst_dir}")

# Example usage

# src_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd1g1'
# dst_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd1g1_sample'

# src_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd1g2'
# dst_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd1g2_sample'
        
src_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd2g1'
dst_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd2g1_sample'

# src_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd2g2'
# dst_dir = DATA_DIR / 'reid' / 'cropped_faces' / 'd2g2_sample'

N = 100 # Number of random images you want to copy

copy_random_pngs(src_dir, dst_dir, N)
