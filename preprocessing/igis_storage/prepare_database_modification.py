import csv

# Fetched data CSV file path
fetched_data_file = 'new_upload_id.csv'

# Injected data CSV file path
injected_data_file = 'injected_data.csv'

# Output file path
output_file = 'updated_data.csv'

# Read the fetched data from the CSV file
fetched_data = []
with open(fetched_data_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        fetched_data.append(row)

# Read the injected data from the CSV file
injected_data = []
with open(injected_data_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        injected_data.append(row)

# Create a dictionary to store the unique key mappings
unique_key_mapping = {}
for row in fetched_data:
    unique_key = (row['FileName'], row['MinX'], row['MaxX'], row['MinY'], row['MaxY'])
    unique_key_mapping[unique_key] = (row['Id'], row['UploadId'])

# Update the Id and UploadId fields in the injected data
updated_data = []
for row in injected_data:
    unique_key = (row['FileName'], row['MinX'], row['MaxX'], row['MinY'], row['MaxY'])
    if unique_key in unique_key_mapping:
        row['Id'], row['UploadId'] = unique_key_mapping[unique_key]
        updated_data.append(row)
    else:
        print(f"Warning: No matching entry found for unique key: {unique_key}")

# Write the updated data to the output CSV file
fieldnames = ['Id', 'UploadId', 'FileName', 'UseForTraining', 'PlantId', 'MinX', 'MaxX', 'MinY', 'MaxY', 'Approved', 'Injected']
with open(output_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_data)

print(f"Updated data saved to {output_file}")