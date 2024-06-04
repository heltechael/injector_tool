import os
import json
import cv2
from datetime import datetime
from tqdm import tqdm

BLACKLIST_EPPO = []  # ["PPPDD", "PPPMM"]
ONLY_APPROVED_IMAGES = True

def create_and_name_directory(output_dir, ONLY_APPROVED_IMAGES):
    """
    Creates a directory with a specific naming format based on the provided parameters.

    Args:
        output_dir (str): Path to the output directory.
        approved_status (str): Status of the annotations (e.g., "Approved", "NotApproved", "M115").
    
    Returns:
        str: Path to the created directory.
    """

    approved_status = "Approved" if ONLY_APPROVED_IMAGES else "NotApproved"
    
    date_str = datetime.now().strftime("%d%b%y").upper()
    dir_name = f"{date_str}_{approved_status}_original_size"
    dir_path = os.path.join(output_dir, dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory: {dir_path} made.")

    return dir_path

def extract_thumbnails(full_images_dir, polydata_dir, output_dir):
    """
    Extracts thumbnails from full images based on polydata files and stores them as PNG files in EPPO-named folders.

    Args:
        full_images_dir (str): Path to the directory containing full images.
        polydata_dir (str): Path to the directory containing polydata files.
        output_dir (str): Path to the output directory where thumbnails will be saved.
    """

    # Count total approved annotations for tqdm progress bar
    total_annotations = 0
    for filename in os.listdir(polydata_dir):
        if filename.endswith(".json"):
            with open(os.path.join(polydata_dir, filename), "r") as f:
                polydata = json.load(f)
                total_annotations += len([ann for ann in polydata if ann["Approved"]])

    # Extract thumbnails
    with tqdm(total=total_annotations, desc="Extracting Thumbnails", unit=" annotations") as pbar:
        for filename in os.listdir(polydata_dir):
            if filename.endswith(".json"):
                upload_id = filename.split("_")[0]
                image_id = filename.split("_")[1]

                full_image_path = os.path.join(full_images_dir, f"{filename[:-5]}")
                polydata_path = os.path.join(polydata_dir, filename)

                with open(polydata_path, "r") as f:
                    polydata = json.load(f)
                image = cv2.imread(full_image_path)

                for annotation in polydata:
                    if ONLY_APPROVED_IMAGES:
                        if not annotation["Approved"]:
                            continue
                
                    eppo_code = annotation["PlantInfo"]["EPPOCode"].strip()
                    if eppo_code in BLACKLIST_EPPO:
                        continue
                    
                    pbar.update(1)

                    thumb_id = annotation["Id"]

                    # Extract bounding box and EPPO code
                    x_min, x_max, y_min, y_max = annotation["MinX"], annotation["MaxX"], annotation["MinY"], annotation["MaxY"]

                    # Extract thumbnail
                    thumbnail = image[y_min : y_max + 1, x_min : x_max + 1]

                    # Create output directory for the eppo
                    output_eppo_dir = os.path.join(output_dir, eppo_code)
                    os.makedirs(output_eppo_dir, exist_ok=True)

                    # Save thumbnail with original filename and coordinates
                    thumbnail_filename = f"{eppo_code}_{upload_id}_{image_id}_{thumb_id}.png"
                    thumbnail_path = os.path.join(output_eppo_dir, thumbnail_filename)
                    cv2.imwrite(thumbnail_path, thumbnail)

full_images_dir = "TrainingData/full_images"
polydata_dir = "TrainingData/polydata"
output_dir = create_and_name_directory("TrainingData/thumbnails_output", ONLY_APPROVED_IMAGES)

extract_thumbnails(full_images_dir, polydata_dir, output_dir)