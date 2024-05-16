"""
import cv2
import numpy as np

class ThumbnailAssessor:
    def __init__(self, config):
        self.config = config
        self.config = config
        self.completeness_threshold = 0.8
        self.centering_threshold = 0.6
        self.background_threshold = 0.7
        self.ratio_min = 0.4
        self.ratio_max = 0.8
        self.edge_threshold = 0.8

    def assess(self, thumbnail_image):
        # Remove the background
        no_background_image = self.remove_background(thumbnail_image)

        # Check for straight edges
        straight_edge_score = self.check_straight_edges(no_background_image)
        completeness_score = self.plant_completeness(thumbnail_image) # MØRK TIL LYS BASICALLY
        centering_score = self.plant_centering(thumbnail_image) # VED IKKE - DECENT
        background_score = self.background_uniformity(thumbnail_image) # MØRK TIL LYS
        ratio_score = self.plant_to_thumbnail_ratio(thumbnail_image)
        edge_score = self.edge_smoothness(thumbnail_image)

        # Combine the scores based on a weighted formula or any other suitable approach
        #assessment_score = straight_edge_score
        #assessment_score = self.combine_scores(completeness_score, centering_score, background_score, ratio_score, edge_score)
        assessment_score = edge_score

        print(f"straight_edge_score: {straight_edge_score}")
        print(f"completeness_score: {completeness_score}")
        print(f"centering_score: {centering_score}")
        print(f"background_score: {background_score}")
        print(f"ratio_score: {ratio_score}")
        print(f"edge_score: {edge_score}")
        print(f"assessment_score: {assessment_score}")
        print("-----------------------------")

        return assessment_score

    def check_background_uniformity(self, thumbnail_image):
        # Convert the thumbnail to grayscale
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Calculate the edge density
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

        background_uniformity_score = 1 - edge_density
        return background_uniformity_score
    
    def remove_background(self, thumbnail_image):
        # Convert the thumbnail to grayscale
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply binary thresholding to create a mask
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply the mask to the original thumbnail image
        no_background_image = cv2.bitwise_and(thumbnail_image, thumbnail_image, mask=mask)

        return no_background_image
    
    def check_straight_edges(self, no_background_image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(no_background_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)

        # Apply Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)

        if lines is not None:
            # Calculate the total length of detected lines
            total_line_length = sum([np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for line in lines for x1, y1, x2, y2 in line])

            # Calculate the score based on the ratio of line length to image perimeter
            image_perimeter = 2 * (no_background_image.shape[0] + no_background_image.shape[1])
            straight_edge_score = total_line_length / image_perimeter
        else:
            straight_edge_score = 0

        return 1 - straight_edge_score
    

    
    def check_plant_completeness(self, thumbnail_image):
        height, width = thumbnail_image.shape[:2]
        edge_pixels = 10  # Number of pixels to consider as edge

        # Check if there are green pixels (plant) at the edges of the thumbnail
        edges = [
            thumbnail_image[0:edge_pixels, :],  # Top edge
            thumbnail_image[-edge_pixels:, :],  # Bottom edge
            thumbnail_image[:, 0:edge_pixels],  # Left edge
            thumbnail_image[:, -edge_pixels:]   # Right edge
        ]                                        

        green_edges = 0
        for edge in edges:
            hsv_edge = cv2.cvtColor(edge, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv_edge, (36, 25, 25), (70, 255, 255))
            green_pixels = cv2.countNonZero(green_mask)
            if green_pixels > 0:
                green_edges += 1

        completeness_score = 1 - (green_edges / 4)
        return completeness_score

    
    def check_single_plant_presence(self, thumbnail_image):
        # Convert the thumbnail to grayscale
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of significant contours (plants)
        num_plants = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"area: {area}")
            if area > 100:  # Adjust this value based on your requirements
                num_plants += 1

        single_plant_score = 1 if num_plants == 1 else 0
        return single_plant_score
    

    def plant_completeness(self, thumbnail_image):
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        height, width = thumbnail_image.shape[:2]
        border_pixels = int(min(height, width) * 0.1)

        border_edges = np.concatenate((
            edges[:border_pixels, :].flatten(),
            edges[-border_pixels:, :].flatten(),
            edges[:, :border_pixels].flatten(),
            edges[:, -border_pixels:].flatten()
        ))

        completeness_score = 1 - (np.sum(border_edges) / (border_edges.shape[0] * 255))
        return completeness_score

    def plant_centering(self, thumbnail_image):
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0

        height, width = thumbnail_image.shape[:2]
        thumbnail_center = (width // 2, height // 2)

        distance = np.sqrt((cx - thumbnail_center[0])**2 + (cy - thumbnail_center[1])**2)
        max_distance = np.sqrt((width // 2)**2 + (height // 2)**2)

        centering_score = 1 - (distance / max_distance)
        return centering_score

    def background_uniformity(self, thumbnail_image):
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        background = cv2.bitwise_and(thumbnail_image, thumbnail_image, mask=cv2.bitwise_not(thresh))
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        uniformity_score = 1 - (np.std(background_gray) / 128)
        return uniformity_score

    def plant_to_thumbnail_ratio(self, thumbnail_image):
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        plant_pixels = cv2.countNonZero(thresh)
        total_pixels = thumbnail_image.shape[0] * thumbnail_image.shape[1]

        ratio = plant_pixels / total_pixels
        ratio_score = 1 - abs(ratio - (self.ratio_min + self.ratio_max) / 2)
        return ratio_score

    def edge_smoothness(self, thumbnail_image):
        gray = cv2.cvtColor(thumbnail_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            smoothness_score = 1 - (len(approx) / len(contour))
        else:
            smoothness_score = 0

        return smoothness_score

    def combine_scores(self, completeness_score, centering_score, background_score, ratio_score, edge_score):
        if (completeness_score >= self.completeness_threshold and
            centering_score >= self.centering_threshold and
            background_score >= self.background_threshold and
            self.ratio_min <= ratio_score <= self.ratio_max and
            edge_score >= self.edge_threshold):
            assessment_score = 1
        else:
            assessment_score = 0

        return assessment_score

        
"""