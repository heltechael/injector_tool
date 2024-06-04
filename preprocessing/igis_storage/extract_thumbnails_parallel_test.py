import os
import csv
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool

BLACKLIST_PLANT_IDS = []  # [145, 162]
BATCH_SIZE = 10

def load_plant_info(csv_file):
    plant_info = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant_id = row["Id"]
            eppo_code = row["EPPOCode"]
            plant_info[plant_id] = eppo_code
    return plant_info

def extract_thumbnails(args):
    full_images_dir, csv_file, output_dir, start_idx, end_idx, plant_info = args

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[start_idx:end_idx]

    for row in tqdm(rows, desc=f"Process {start_idx}-{end_idx}", unit=" thumbnails"):
        upload_id = row["UploadId"]
        image_id = row["ImageId"]
        file_name = row["FileName"]
        plant_id = row["PlantId"]

        if int(plant_id) in BLACKLIST_PLANT_IDS:
            continue

        eppo_code = plant_info.get(plant_id, "unknown")
        thumb_id = f"{upload_id}_{image_id}"

        # Extract bounding box
        x_min, x_max, y_min, y_max = int(row["MinX"]), int(row["MaxX"]), int(row["MinY"]), int(row["MaxY"])

        # Load the full image
        full_image_path = os.path.join(full_images_dir, upload_id, file_name)
        if not os.path.exists(full_image_path):
            print(f"Error: Image file not found - {full_image_path}")
            continue

        with open(full_image_path, "rb") as f:
            image_data = np.frombuffer(f.read(), np.uint8)

        # Determine the image format based on file extension
        _, file_extension = os.path.splitext(file_name)
        if file_extension.lower() == ".png":
            image = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Error: Failed to load image - {full_image_path}")
            continue

        # Check if the bounding box coordinates are valid
        if x_min < 0 or y_min < 0 or x_max >= image.shape[1] or y_max >= image.shape[0] or x_min >= x_max or y_min >= y_max:
            print(f"Error: Invalid bounding box coordinates - {full_image_path}")
            print(f"Bounding box: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            print(f"Image shape: {image.shape}")
            continue

        # Extract thumbnail
        thumbnail = image[y_min : y_max + 1, x_min : x_max + 1]

        # Create output directory for the EPPO code
        output_eppo_dir = os.path.join(output_dir, eppo_code)
        os.makedirs(output_eppo_dir, exist_ok=True)

        # Save thumbnail with original filename and coordinates
        thumbnail_filename = f"{eppo_code}_{thumb_id}{file_extension}"
        thumbnail_path = os.path.join(output_eppo_dir, thumbnail_filename)
        _, thumbnail_data = cv2.imencode(file_extension, thumbnail)
        with open(thumbnail_path, "wb") as f:
            f.write(thumbnail_data)

def extract_thumbnails_parallel(full_images_dir, csv_file, output_dir, num_processes, plant_info_file):

    plant_info = load_plant_info(plant_info_file)

    # Count total annotations
    with open(csv_file, "r") as f:
        total_annotations = sum(1 for _ in csv.DictReader(f))

    # Create argument list for each process
    num_annotations_per_process = total_annotations // num_processes
    args_list = []
    for i in range(num_processes):
        start_idx = i * num_annotations_per_process
        end_idx = (i + 1) * num_annotations_per_process if i < num_processes - 1 else total_annotations
        args_list.append((full_images_dir, csv_file, output_dir, start_idx, end_idx, plant_info))

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Distribute the work among the processes
        pool.map(extract_thumbnails, args_list)

full_images_dir = "/mnt/rwmdata/michael-data/"
csv_file = "csv_data_IGIS/all_thumbs_from_unapproved_subset.csv"
output_dir = "/mnt/rwmdata/michael-data/thumbnails"
plant_info_file = "csv_data_IGIS/plant_info_IGIS.csv"

num_processes = 10 # Adjust based on your system's capabilities

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory: {output_dir} made.")

print(f"Training started with batch size: {BATCH_SIZE} and {num_processes} processes.")

extract_thumbnails_parallel(full_images_dir, csv_file, output_dir, num_processes, plant_info_file)
