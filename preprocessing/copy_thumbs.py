import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool

eppo_codes2 = [
    "1POLF", "1SONG", "GALAP", "EQUAR", "1MATG", "FUMOF", "STEME",
    "VIOAR", "PAPRH", "GERMO", "CHEAL", "SOLNI", "1LUPG", "FAGES", "1RUMG",
    "TAROF"
]

eppo_codes = [
    "1SONG", "EQUAR", "1MATG", "FUMOF", "GERMO", "SOLNI", "1LUPG", "1RUMG",
    "TAROF"
]

max_files = 3000

source_dir = "/mnt/network_folder/main-data/michael-data/training_thumbnails/"
dest_dir = "."

num_processes = 10

def copy_files(eppo_code):
    subdir_path = os.path.join(source_dir, eppo_code)

    if os.path.exists(subdir_path):
        files = os.listdir(subdir_path)
        files = files[:max_files]

        dest_subdir = os.path.join(dest_dir, eppo_code)
        os.makedirs(dest_subdir, exist_ok=True)

        for i in tqdm(range(0, len(files), 10), desc=eppo_code, unit="batch"):
            batch = files[i:i+10]
            for file_name in batch:
                source_file = os.path.join(subdir_path, file_name)
                dest_file = os.path.join(dest_subdir, file_name)
                shutil.copy2(source_file, dest_file)

        return f"Copied {len(files)} files from {eppo_code} to {dest_subdir}"
    else:
        return f"Subdirectory {eppo_code} does not exist"

if __name__ == "__main__":
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(copy_files, eppo_codes), total=len(eppo_codes), desc="Overall Progress"))

    for result in results:
        print(result)