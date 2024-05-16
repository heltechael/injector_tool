import csv
from collections import defaultdict

class CSVUtils:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.csv_data = self.load_csv_file()
        self.indexed_data = self.index_csv_data()

    def load_csv_file(self):
        data = []
        with open(self.csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    
    def filter_csv_data(self, used_full_images):
        filtered_data = []
        for image_data in used_full_images:
            upload_id, image_id, filename = image_data.filename.split('_', 2)
            key = f"{upload_id}_{image_id}_{filename}"
            filtered_data.extend(self.indexed_data[key])
        return filtered_data

    def index_csv_data(self):
        index = defaultdict(list)
        for row in self.csv_data:
            key = f"{row['UploadId']}_{row['ImageId']}_{row['FileName']}"
            index[key].append(row)
        return index

    def get_bounding_boxes(self, upload_id, image_id, filename):
        key = f"{upload_id}_{image_id}_{filename}"
        bounding_boxes = []
        for row in self.indexed_data[key]:
            minX = int(row['MinX'])
            maxX = int(row['MaxX'])
            minY = int(row['MinY'])
            maxY = int(row['MaxY'])
            bounding_boxes.append((minX, maxX, minY, maxY))
        return bounding_boxes

    def add_injected_bounding_box(self, new_annotation):
        self.csv_data.append(new_annotation)
        key = f"{new_annotation['UploadId']}_{new_annotation['ImageId']}_{new_annotation['FileName']}"
        self.indexed_data[key].append(new_annotation)
