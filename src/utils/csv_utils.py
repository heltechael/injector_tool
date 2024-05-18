import csv
from collections import defaultdict

class CSVUtils:
    def __init__(self, csv_path, eppo_to_plant_id_translator):
        self.csv_path = csv_path
        self.eppo_to_plant_id_translator = eppo_to_plant_id_translator
        self.csv_data = self.load_csv_file()
        self.indexed_data = self.index_csv_data()

    def load_csv_file(self):
        data = []
        with open(self.csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Ensure Injected field is present and set to False for original data
                row['Injected'] = 'False'
                data.append(row)
        return data

    def filter_csv_data(self, used_full_images):
        filtered_data = []
        for image_data in used_full_images:
            upload_id = image_data.uploadid
            filename = image_data.filename
            key = f"{upload_id}_{filename}"
            filtered_data.extend(self.indexed_data[key])
        return filtered_data

    def index_csv_data(self):
        index = defaultdict(list)
        for row in self.csv_data:
            key = f"{row['UploadId']}_{row['FileName']}"
            index[key].append(row)
        return index

    def get_bounding_boxes(self, upload_id, filename):
        key = f"{upload_id}_{filename}"
        bounding_boxes = []
        for row in self.indexed_data[key]:
            minX = int(row['MinX'])
            maxX = int(row['MaxX'])
            minY = int(row['MinY'])
            maxY = int(row['MaxY'])
            bounding_boxes.append((minX, maxX, minY, maxY))
        return bounding_boxes

    def add_injected_bounding_box(self, new_annotation):
        eppo_code = new_annotation['PlantId']
        plant_id = self.eppo_to_plant_id_translator.translate(eppo_code)
        new_annotation['PlantId'] = plant_id if plant_id else "NEJ"
        new_annotation['Injected'] = 'True'
        self.csv_data.append(new_annotation)
        key = f"{new_annotation['UploadId']}_{new_annotation['FileName']}"
        self.indexed_data[key].append(new_annotation)

    def save_filtered_csv_file(self, csv_path, csv_data):
        # Define the fields to keep in the output CSV file
        fields_to_keep = [
            'UploadId', 'FileName', 'UseForTraining', 
            'PlantId', 'MinX', 'MaxX', 'MinY', 'MaxY', 
            'Approved', 'Injected'
        ]

        # Filter the CSV data to only include the specified fields
        filtered_data = []
        for row in csv_data:
            filtered_row = {field: row.get(field, '') for field in fields_to_keep}
            filtered_data.append(filtered_row)

        # Save the filtered data to a new CSV file
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields_to_keep)
            writer.writeheader()
            writer.writerows(filtered_data)

    def save_updated_csv_file(self, csv_path, csv_data):
        # Define the fields to keep in the output CSV file
        fields_to_keep = [
            'UploadId', 'FileName', 'UseForTraining', 
            'PlantId', 'MinX', 'MaxX', 'MinY', 'MaxY', 
            'Approved', 'Injected'
        ]

        # Filter the CSV data to only include the specified fields
        filtered_data = []
        for row in csv_data:
            filtered_row = {field: row.get(field, '') for field in fields_to_keep}
            filtered_data.append(filtered_row)

        # Save the filtered data to a new CSV file
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields_to_keep)
            writer.writeheader()
            writer.writerows(filtered_data)
