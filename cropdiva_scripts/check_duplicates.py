import os
import hashlib
from tqdm import tqdm

# quick hashing with SHA-1 based on file size and first+last 1024 bytes
def quick_hash(filepath):
    size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        first_part = f.read(1024)
        f.seek(-1024, os.SEEK_END)
        last_part = f.read(1024)
    return hashlib.sha1(first_part + last_part + str(size).encode()).hexdigest()

# full hashing with SHA-1
def full_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha1()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def find_duplicates(parent_folder):
    quick_hash_dict = {}
    full_hash_dict = {}
    duplicates = []
    folders = [os.path.join(parent_folder, folder) for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
    
    for folder in tqdm(folders, desc="Checking Folders"):
        for dirpath, _, filenames in os.walk(folder):
            plant = dirpath.replace(parent_folder, '')
            for filename in tqdm(filenames, desc=f"Analyzing: {plant}"):
                # Skip non-image files
                if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
                    continue
                file_path = os.path.join(dirpath, filename)
                q_hash = quick_hash(file_path)
                if q_hash in quick_hash_dict:
                    # Perform full hash check only if quick hash matches
                    f_hash = full_hash(file_path)
                    if f_hash in full_hash_dict:
                        duplicates.append((full_hash_dict[f_hash], file_path))
                    else:
                        full_hash_dict[f_hash] = file_path
                else:
                    quick_hash_dict[q_hash] = file_path
                    # add to full hash dict to avoid recalculating
                    full_hash_dict[full_hash(file_path)] = file_path

    return duplicates

parent_folder = '../TrainingData/08DEC24_Approved_scale224px_min50px_padding/'  # Update this to your actual parent folder path
duplicates = find_duplicates(parent_folder)

if duplicates:
    print(f"Found {len(duplicates)} duplicates.")
    for dup in duplicates:
        print(f"Duplicate pair: {dup[0]} and {dup[1]}")
else:
    print("No duplicates found.")
