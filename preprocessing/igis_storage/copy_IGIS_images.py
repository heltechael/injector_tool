import os
import csv
import shutil
import time
from tqdm import tqdm

# Set the paths
csv_file = 'csv_data_IGIS/all_thumbs_from_unapproved.csv'  # Replace with your CSV file name
network_folder = '/mnt/rwmdata/'
output_directory = '/mnt/rwmdata/michael-data/'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read the CSV file and count the total number of rows
with open(csv_file, 'r') as file:
    total_rows = sum(1 for _ in file) - 1  # Subtract 1 for the header row

# Read the CSV file again
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    processed_images = set()

    # Create a progress bar
    progress_bar = tqdm(total=total_rows, unit='images', desc='Copying images')

    sleep_count = 0
    for row in csv_reader:
        upload_id = row['UploadId']
        file_name = row['FileName']

        # Check if the image has already been processed
        if (upload_id, file_name) in processed_images:
            progress_bar.update(1)  # Update the progress bar
            continue

        # Construct the source and destination paths
        source_path = os.path.join(network_folder, upload_id, file_name)
        destination_folder = os.path.join(output_directory, upload_id)
        destination_path = os.path.join(destination_folder, file_name)

        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        # Copy the image file from the network folder to the local directory
        shutil.copy2(source_path, destination_path)

        # Mark the image as processed
        processed_images.add((upload_id, file_name))

        if sleep_count>30:
            time.sleep(1)
            sleep_count = 0
            print(f"Sleeping to give network a break.")

        # Update the progress bar
        progress_bar.update(1)
        sleep_count += 1

    # Close the progress bar
    progress_bar.close()

print("Image files copied successfully.")