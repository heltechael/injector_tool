import json
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

class GridUtils:
    def __init__(self, config):
        self.config = config

    def get_bounding_boxes_list(self, polydata):
        bounding_boxes = []
        for annotation in polydata:
            if 'MinX' in annotation and 'MaxX' in annotation and 'MinY' in annotation and 'MaxY' in annotation:
                bounding_boxes.append((annotation['MinX'], annotation['MaxX'], annotation['MinY'], annotation['MaxY']))
        return bounding_boxes
    
    def find_unoccupied_cells_matrix(self, bounding_boxes, image_width, image_height, cell_size):
        # Calculate the number of cells in the grid
        num_cells_x = image_width // cell_size
        num_cells_y = image_height // cell_size

        # Create a 2D matrix representing the grid
        grid = np.zeros((num_cells_x, num_cells_y), dtype=np.uint8)

        # Mark the occupied cells in the grid
        for minX, maxX, minY, maxY in bounding_boxes:
            start_cell_x = minX // cell_size
            end_cell_x = maxX // cell_size
            start_cell_y = minY // cell_size
            end_cell_y = maxY // cell_size
            grid[start_cell_x:end_cell_x+1, start_cell_y:end_cell_y+1] = 1

        return grid
    
    def update_unoccupied_cells_matrix(self, grid, bbox, cell_size):
        minX, maxX, minY, maxY = bbox

        # Mark the occupied cells in the grid
        start_cell_x = minX // cell_size
        end_cell_x = maxX // cell_size
        start_cell_y = minY // cell_size
        end_cell_y = maxY // cell_size
        grid[start_cell_x:end_cell_x+1, start_cell_y:end_cell_y+1] = 1

        return grid
    
    def find_viable_position(self, thumbnail_width, thumbnail_height, unoccupied_cells_matrix, cell_size, random_start=True):
        cells_num_x = unoccupied_cells_matrix.shape[0]
        cells_num_y = unoccupied_cells_matrix.shape[1]
        
        # Calculate the number of cells required for the thumbnail
        thumb_cells_width = (thumbnail_width + cell_size - 1) // cell_size
        thumb_cells_height = (thumbnail_height + cell_size - 1) // cell_size

        # Start "sliding window" at random place 
        if random_start:
            start_cell_x = random.randint(0, cells_num_x)
            start_cell_y = random.randint(0, cells_num_y)

            # Find the first unoccupied position that fits the thumbnail
            for i in range(cells_num_x - thumb_cells_width + 1):
                for j in range(cells_num_y - thumb_cells_height + 1):
                    
                    x_pointer = (i + start_cell_x) % cells_num_x
                    y_pointer = (j + start_cell_y) % cells_num_y

                    cell_start_x = x_pointer
                    cell_end_x = x_pointer + thumb_cells_width
                    cell_start_y = y_pointer
                    cell_end_y = y_pointer + thumb_cells_height

                    if ((x_pointer*cell_size + thumbnail_width+1) >= (cells_num_x+1)*cell_size):
                        continue

                    if ((y_pointer*cell_size + thumbnail_height+1) >= (cells_num_y+1)*cell_size):
                        continue

                    # Check if the current window is unoccupied
                    if np.sum(unoccupied_cells_matrix[cell_start_x:cell_end_x, cell_start_y:cell_end_y]) == 0:
                        return (cell_start_x * cell_size, cell_start_y * cell_size)
                    
        else:
            for i in range(unoccupied_cells_matrix.shape[0] - thumb_cells_width + 1):
                for j in range(unoccupied_cells_matrix.shape[1] - thumb_cells_height + 1):
                    if np.sum(unoccupied_cells_matrix[i:i+thumb_cells_width, j:j+thumb_cells_height]) == 0:
                        return (i * cell_size, j * cell_size)

        return None
    
