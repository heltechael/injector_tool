import csv

# Specify the input and output file paths
input_file = 'filtered_annotations.csv'
output_file = 'new_upload_id_injected.csv'

# Specify the fields to remove
fields_to_remove = ['PolyData', 'AnnotationId', 'ImageId', 'AnnotationScore', 'BrushSize', 'UserId', 'AnnotationModelId', 'ClassificationModelId', 'GrowthStage', 'BrushSizePadded', 'CreationTime', 'IsTemporary', 'ClassificationScore']

# Read the input CSV file
with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames

    # Remove the specified fields from the fieldnames
    fieldnames = [field for field in fieldnames if field not in fields_to_remove]

    # Write the updated data to the output CSV file
    with open(output_file, 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Remove the specified fields from each row
            for field in fields_to_remove:
                del row[field]
            writer.writerow(row)

print("CSV file updated successfully.")