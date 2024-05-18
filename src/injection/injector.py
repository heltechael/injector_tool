import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from src.utils.thumbnail_utils import ThumbnailUtils
from src.utils.grid_utils import GridUtils
from src.utils.csv_utils import CSVUtils
from src.data_loader.data_loader import DataLoader

class Injector:
    def __init__(self, config, data_loader, thumbdata, csv_path, eppo_to_plant_id_translator) -> None:
        self.config = config
        self.dataLoader = data_loader
        self.thumbdata = thumbdata
        self.csv_path = csv_path
        self.thumbnailUtils = ThumbnailUtils(config)
        self.gridUtils = GridUtils(config)
        self.csvUtils = CSVUtils(csv_path, eppo_to_plant_id_translator)

    def select_random_thumbnails(self, number_of_thumbnails_to_return=1):
        selected_thumbnails = []

        if len(self.thumbdata) < number_of_thumbnails_to_return:
            number_of_thumbnails_to_return = len(self.thumbdata)p

        for i in range(number_of_thumbnails_to_return):
            selected_thumbnail = random.choice(self.thumbdata)
            selected_thumbnails.append(selected_thumbnail)
            self.thumbdata.remove(selected_thumbnail)
        
        return selected_thumbnails

    def inject_bbox(self, full_image, bbox_image, position):
        x, y = position
        bbox_height, bbox_width, _ = bbox_image.shape
        
        # Extract the region of interest (ROI) from the full image
        roi = full_image[y:y+bbox_height, x:x+bbox_width]

        if self.config.get("REMOVE_THUMBNAIL_BACKGROUND"):
            bbox_image = self.thumbnailUtils.remove_background(bbox_image)
            bbox_image = self.thumbnailUtils.blend_thumbnail_edges(bbox_image)

        # Overlay the thumbnail on the ROI
        blended_roi = self.thumbnailUtils.overlay_thumbnail(roi, bbox_image)

        # Update the full image with the blended ROI
        full_image[y:y+bbox_height, x:x+bbox_width] = blended_roi

        bbox = (x, x+bbox_width, y, y+bbox_height)

        return full_image, bbox

    def inject_thumbnails_into_single_full_image(self, full_image, image_data, num_injections):
        MAX_INJECTIONS_PER_IMAGE = self.config.get("MAX_INJECTIONS_PER_IMAGE")
        CELL_SIZE = self.config.get("CELL_SIZE")
        
        # Copy full image to create image viable for modification
        injected_image = full_image.copy()
        full_image_height, full_image_width, _ = full_image.shape

        # Select random thumbnails from selected thumbnails to ensure no order bias
        thumbnails_to_inject = min(num_injections, MAX_INJECTIONS_PER_IMAGE)
        selected_thumbnails = self.select_random_thumbnails(thumbnails_to_inject)
        num_injections = len(selected_thumbnails)

        # Compute unoccupied locations in full image
        output_images_dir = self.config.get('output_dir')
        upload_id = image_data.uploadid
        filename = image_data.filename
        bounding_boxes = self.csvUtils.get_bounding_boxes(upload_id, filename)
        unoccupied_cells_matrix = self.gridUtils.find_unoccupied_cells_matrix(bounding_boxes, full_image_width, full_image_height, CELL_SIZE)

        # Inject thumbnails into the full image
        for i in range(num_injections):

            # Choose a thumbnail
            thumbdata = selected_thumbnails[i]
            thumbnail = self.dataLoader.load_image(thumbdata.path)
            thumb_height, thumb_width, _ = thumbnail.shape

            # Find viable position for inserting thumbnail
            decent_position = self.gridUtils.find_viable_position(thumb_width, thumb_height, unoccupied_cells_matrix, CELL_SIZE)
            
            # Insert thumbnail at position (x,y)
            if decent_position:
                x, y = decent_position
                try:
                    injected_image, bbox = self.inject_bbox(injected_image, thumbnail, (x, y))
                    bounding_boxes.append(bbox)
                    unoccupied_cells_matrix = self.gridUtils.update_unoccupied_cells_matrix(unoccupied_cells_matrix, bbox, CELL_SIZE)
                    
                    # Create a new annotation for the injected thumbnail
                    new_annotation = {
                        'UploadId': upload_id,
                        'FileName': filename,
                        'UseForTraining': 'True',
                        'PlantId': thumbdata.eppo,
                        'MinX': str(x),
                        'MaxX': str(x + thumb_width),
                        'MinY': str(y),
                        'MaxY': str(y + thumb_height),
                        'Approved': 'True',
                        'Injected': 'True'
                    }
                    self.csvUtils.add_injected_bounding_box(new_annotation)
                except ValueError as e:
                    print(f"Error injecting thumbnail: {e}")
                    self.thumbdata.append(thumbdata)
            
            # If no suitable position found for the thumbnail -> append thumbdata back in stack
            else:
                self.thumbdata.append(thumbdata)

        # Store debug full image with grid and bounding boxes
        output_bounding_boxes_dir = self.config.get('output_bounding_boxes_dir')
        if self.config.get('DEBUG'):
            self.draw_bounding_boxes_on_image_with_grids(injected_image, bounding_boxes, unoccupied_cells_matrix, CELL_SIZE, output_bounding_boxes_dir, filename)
        return injected_image, bounding_boxes
    
    def inject_thumbnails_into_n_full_images(self, num_full_images, max_injections, output_images_dir):
        injected_images = []
        used_full_images = []

        for _ in tqdm(range(num_full_images), desc="Injecting thumbnails"):
            image_data = self.dataLoader.get_random_full_image()
            full_image = self.dataLoader.load_image(image_data.path)
            injected_image, _ = self.inject_thumbnails_into_single_full_image(full_image, image_data, max_injections)
            
            # Store the injected image locally
            self.dataLoader.store_image(injected_image, f"{output_images_dir}/{os.path.splitext(image_data.filename)[0]}.jpg")
            
            injected_images.append((injected_image, image_data))
            used_full_images.append(image_data)

        # Filter the CSV data based on the used full images
        filtered_csv_data = self.csvUtils.filter_csv_data(used_full_images)

        # Save the filtered CSV file
        output_csv_file = self.config.get('output_csv_file')
        self.csvUtils.save_filtered_csv_file(output_csv_file, filtered_csv_data)

        return injected_images

    def draw_bounding_boxes_on_image_with_grids(self, image, bounding_boxes, unoccupied_cells_matrix, cell_size, output_bounding_boxes_dir, filename):
        # Create a copy of the image to avoid modifying the original
        image_with_boxes = image.copy()

        # Draw the grid lines
        num_cells_x, num_cells_y = unoccupied_cells_matrix.shape

        for i in range(num_cells_y + 1):
            cv2.line(image_with_boxes, (0, i * cell_size), (image.shape[1], i * cell_size), (128, 128, 128), 1)
        for j in range(num_cells_x + 1):
            cv2.line(image_with_boxes, (j * cell_size, 0), (j * cell_size, image.shape[0]), (128, 128, 128), 1)

        # Iterate over the bounding boxes and draw them
        for minX, maxX, minY, maxY in bounding_boxes:
            cv2.rectangle(image_with_boxes, (minX, minY), (maxX, maxY), (0, 255, 0), 2)

        # Store the image with bounding boxes and grid using the specified function
        self.dataLoader.store_image(image_with_boxes, f"{output_bounding_boxes_dir}/{filename}.jpg")
