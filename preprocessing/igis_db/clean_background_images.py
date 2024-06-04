import csv

input_file = 'filtered_annotations_upload1.csv'
output_file = 'filtered_annotations_upload1_cleaned.csv'

with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    writer.writeheader()
    
    for row in reader:
        if row['Injected'] != 'False':
            writer.writerow(row)

print(f"Filtered data has been written to {output_file}")
