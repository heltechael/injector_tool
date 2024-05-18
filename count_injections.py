import csv
from collections import defaultdict

# Specify the input CSV file path
input_file = 'output/injection20/filtered_annotations.csv'

# Initialize a dictionary to store the counts for each PlantId
counts_by_plantid = defaultdict(int)

# Read the input CSV file
with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    
    # Iterate over each row in the input CSV file
    for row in reader:
        # Check if the 'Injected' field is 'True'
        if row['Injected'] == 'True':
            # Get the PlantId for the current row
            plantid = row['PlantId']
            
            # Increment the count for the corresponding PlantId
            counts_by_plantid[plantid] += 1

# Print the counts for each PlantId
for plantid, count in counts_by_plantid.items():
    print(f"PlantId {plantid}: {count} counts")