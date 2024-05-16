import os
import cv2
import csv
from tqdm import tqdm
from collections import namedtuple, defaultdict

FullImageData = namedtuple('ImageData', ['path', 'filename', 'uploadid'])
ThumbnailData = namedtuple('ThumbData', ['path', 'filename', 'eppo'])

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.image_cache = {}

    def load_csv_file(self, csv_path):
        data = []
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    
    def save_filtered_csv_file(self, csv_path, csv_data):
        fieldnames = list(csv_data[0].keys())
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

    def save_updated_csv_file(self, csv_path, csv_data):
        fieldnames = list(csv_data[0].keys())
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

    def load_image(self, image_path):
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            self.image_cache[image_path] = image
            return image
        
    def store_image(self, image, image_path):
        if cv2.imwrite(image_path, image):
            print(f'Image saved success: {image_path}')
        else:
            print(f'Image write failed: {image_path}')
        
    def load_thumbnails(self):
        thumbnails_dir = self.config.get('thumbnails_dir')
        thumbnails_folder = self.config.get('thumbnails_folder')
        thumbnails_dir = os.path.join(thumbnails_dir, thumbnails_folder)

        if not os.path.exists(thumbnails_dir):
            raise FileNotFoundError(f"Thumbnails directory not found: {thumbnails_dir}")

        thumbnails_by_class = defaultdict(list)
        thumbnail_cache = {}

        for class_dir in os.listdir(thumbnails_dir):
            class_path = os.path.join(thumbnails_dir, class_dir)
            if os.path.isdir(class_path):
                for thumbnail_file in os.listdir(class_path):
                    thumbnail_path = os.path.join(class_path, thumbnail_file)
                    thumbnail_data = ThumbnailData(thumbnail_path, thumbnail_file, class_dir)
                    thumbnails_by_class[class_dir].append(thumbnail_data)

        def load_thumbnail(thumbnail_data):
            if thumbnail_data.path in thumbnail_cache:
                return thumbnail_cache[thumbnail_data.path]
            else:
                thumbnail = self.load_image(thumbnail_data.path)
                thumbnail_cache[thumbnail_data.path] = thumbnail
                return thumbnail

        return thumbnails_by_class, load_thumbnail

    def load_full_images(self):
        full_images_dir = self.config.get('full_images_test_dir')
        full_images = []

        for upload_id in os.listdir(full_images_dir):
            upload_dir = os.path.join(full_images_dir, upload_id)
            if os.path.isdir(upload_dir):
                for image_file in os.listdir(upload_dir):
                    image_path = os.path.join(upload_dir, image_file)
                    image_data = FullImageData(image_path, image_file, upload_id)
                    full_images.append(image_data)

        return full_images
