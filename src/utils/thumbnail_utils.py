import cv2
import numpy as np
import random

class ThumbnailUtils:
    def __init__(self, config):
        self.config = config

    def remove_background(self, bbox_image):
        # Convert the image to float32 for calculations
        bbox_image = bbox_image.astype(np.float32)

        # Calculate ExG (Excess Green)
        ExG = 2 * bbox_image[:, :, 1] - bbox_image[:, :, 2] - bbox_image[:, :, 0]

        # Calculate ExR (Excess Red)
        ExR = 1.4 * bbox_image[:, :, 2] - bbox_image[:, :, 1]

        # Create a binary mask based on the condition ExG - ExR >= 0
        mask = (ExG - ExR >= 0).astype(np.uint8) * 255

        # Apply morphological operations to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply the mask to the original image to remove the background
        result = cv2.bitwise_and(bbox_image.astype(np.uint8), bbox_image.astype(np.uint8), mask=mask)

        return result
    
    def create_bbox_mask(self, bbox_image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to create a binary mask
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        return mask
    
    def blend_thumbnail_edges(self, thumbnail, border_width=5, opacity=0.5):
        # Create a mask for the thumbnail
        mask = self.create_bbox_mask(thumbnail)

        # Create a BGRA version of the thumbnail with transparent background
        thumbnail_bgra = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2BGRA)
        thumbnail_bgra[:, :, 3] = mask

        # Reduce the opacity of the outer pixels
        thumbnail_bgra[:border_width, :, 3] = thumbnail_bgra[:border_width, :, 3] * opacity
        thumbnail_bgra[-border_width:, :, 3] = thumbnail_bgra[-border_width:, :, 3] * opacity
        thumbnail_bgra[:, :border_width, 3] = thumbnail_bgra[:, :border_width, 3] * opacity
        thumbnail_bgra[:, -border_width:, 3] = thumbnail_bgra[:, -border_width:, 3] * opacity

        return thumbnail_bgra

        # Convert the blended thumbnail back to BGR
        #blended_thumbnail = cv2.cvtColor(thumbnail_bgra, cv2.COLOR_BGRA2BGR)

        #return blended_thumbnail
    
    def overlay_thumbnail(self, roi, thumbnail):
        # Create a BGRA version of the ROI
        roi_bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)

        # Create a BGRA version of the thumbnail
        thumbnail_bgra = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2BGRA)

        # Overlay the thumbnail on the ROI
        blended_roi = np.where(thumbnail_bgra[..., 3:] > 0, thumbnail_bgra, roi_bgra)

        # Convert the blended ROI back to BGR
        blended_roi = cv2.cvtColor(blended_roi, cv2.COLOR_BGRA2BGR)

        return blended_roi
