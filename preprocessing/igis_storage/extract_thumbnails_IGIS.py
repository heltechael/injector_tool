import os
import csv
import cv2
from datetime import datetime
from tqdm import tqdm

BLACKLIST_PLANT_IDS = []  # [145, 162]

def create_and_name_directory(output_dir):
    """
    Creates a directory with a specific naming format based on the current date.

    Args:
        output_dir (str): Path to the output directory.
    
    Returns:
        str: Path to the created directory.
    """
    
    date_str = datetime.now().strftime("%d%b%y").upper()
    dir_name = f"{date_str}all_approved_thumbnails_from_unapproved_full_images"
    dir_path = os.path.join(output_dir, dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory: {dir_path} made.")

    return dir_path

def extract_thumbnails(full_images_dir, csv_file, output_dir):
    """
    Extracts thumbnails from full images based on the CSV file and stores them as PNG files in PlantId-named folders.

    Args:
        full_images_dir (str): Path to the directory containing full images.
        csv_file (str): Path to the CSV file containing thumbnail information.
        output_dir (str): Path to the output directory where thumbnails will be saved.
    """

    # Count total annotations for tqdm progress bar
    total_annotations = sum(1 for _ in csv.DictReader(open(csv_file)))

    # Extract thumbnails
    with tqdm(total=total_annotations, desc="Extracting Thumbnails", unit=" annotations") as pbar:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):

                upload_id = row["UploadId"]
                image_id = row["ImageId"]
                file_name = row["FileName"]
                plant_id = row["PlantId"]

                if int(plant_id) in BLACKLIST_PLANT_IDS:
                    continue
                
                pbar.update(1)

                thumb_id = f"{upload_id}_{image_id}"

                # Extract bounding box
                x_min, x_max, y_min, y_max = int(row["MinX"]), int(row["MaxX"]), int(row["MinY"]), int(row["MaxY"])

                # Load the full image
                full_image_path = os.path.join(full_images_dir, upload_id, file_name)
                image = cv2.imread(full_image_path)

                # Extract thumbnail
                thumbnail = image[y_min : y_max + 1, x_min : x_max + 1]

                # Create output directory for the plant id
                output_plant_dir = os.path.join(output_dir, plant_id)
                os.makedirs(output_plant_dir, exist_ok=True)

                # Save thumbnail with original filename and coordinates
                thumbnail_filename = f"{plant_id}_{thumb_id}_{i}.png"
                thumbnail_path = os.path.join(output_plant_dir, thumbnail_filename)
                cv2.imwrite(thumbnail_path, thumbnail)

full_images_dir = "/mnt/rwmdata/michael-data/"
csv_file = "csv_data_IGIS/all_thumbs_from_unapproved_subset.csv"
output_dir = "/mnt/rwmdata/michael-data/thumbnails"

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory: {output_dir} made.")

extract_thumbnails(full_images_dir, csv_file, output_dir)