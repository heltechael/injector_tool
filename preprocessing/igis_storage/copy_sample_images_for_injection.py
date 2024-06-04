import os
import csv
import shutil
import time
from tqdm import tqdm
import random

# Set the paths
csv_file = 'csv_data_IGIS/all_training_thumbnails.csv'  # Replace with your CSV file name
network_folder = '/mnt/rwmdata/'
output_directory = '/mnt/rwmdata/michael-data/sample_full_images_new/'
output_csv_file = '/mnt/rwmdata/michael-data/sample_full_images_new/samples_images.csv'

SAMPLE_SIZE = 5000

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read the CSV file and extract all rows
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    rows = list(csv_reader)

# Extract unique image identifiers (UploadId, FileName)
image_identifiers = list(set((row['UploadId'], row['FileName']) for row in rows))

# Randomly sample 5000 image identifiers
sampled_identifiers = random.sample(image_identifiers, min(5000, len(image_identifiers)))

# Open the output CSV file
with open(output_csv_file, 'w', newline='') as output_file:
    csv_writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
    csv_writer.writeheader()

    # Create a progress bar
    progress_bar = tqdm(total=len(sampled_identifiers), unit='images', desc='Copying images')

    #sleep_count = 0
    for upload_id, file_name in sampled_identifiers:
        # Construct the source and destination paths
        source_path = os.path.join(network_folder, upload_id, file_name)
        destination_folder = os.path.join(output_directory, upload_id)
        destination_path = os.path.join(destination_folder, file_name)

        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        # Copy the image file from the network folder to the local directory
        shutil.copy2(source_path, destination_path)

        # Write the corresponding rows to the output CSV file
        for row in rows:
            if row['UploadId'] == upload_id and row['FileName'] == file_name:
                csv_writer.writerow(row)

        """
        if sleep_count > 30:
            time.sleep(1)
            sleep_count = 0
            print("Sleeping to give network a break.")

        sleep_count += 1
        """

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

print("Image files copied successfully.")