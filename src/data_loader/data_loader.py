import os
import cv2
import csv
import glob
import random
from tqdm import tqdm
from collections import namedtuple, defaultdict

FullImageData = namedtuple('ImageData', ['path', 'filename', 'uploadid'])
ThumbnailData = namedtuple('ThumbData', ['path', 'filename', 'eppo'])

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.image_cache = {}
        self.full_images = self.load_full_images()

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
        cv2.imwrite(image_path, image)

    def load_thumbnails(self, thumbnail_classes):
        thumbnails_dir = self.config.get('thumbnails_dir')
        thumbnails_folder = self.config.get('thumbnails_folder')
        thumbnails_dir = os.path.join(thumbnails_dir, thumbnails_folder)

        if not os.path.exists(thumbnails_dir):
            raise FileNotFoundError(f"Thumbnails directory not found: {thumbnails_dir}")

        thumbnails_by_class = defaultdict(list)
        thumbnail_cache = {}

        for class_dir in thumbnail_classes:
            thumbnail_paths = glob.glob(os.path.join(thumbnails_dir, class_dir, "*"))
            for thumbnail_path in tqdm(thumbnail_paths, desc=f"Loading thumbnails for class {class_dir}"):
                thumbnail_file = os.path.basename(thumbnail_path)
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
        full_images_dir = self.config.get('full_images_dir')
        full_images = []

        for upload_id in tqdm(os.listdir(full_images_dir), desc="Loading full images UploadId's"):
            upload_dir = os.path.join(full_images_dir, upload_id)
            if os.path.isdir(upload_dir):
                for image_file in os.listdir(upload_dir):
                    image_path = os.path.join(upload_dir, image_file)
                    image_data = FullImageData(image_path, image_file, upload_id)
                    full_images.append(image_data)

        return full_images
    
    def get_random_full_image(self):
        return random.choice(self.full_images)
