import cv2
import numpy as np

def remove_background(bbox_image):
    B, G, R = cv2.split(bbox_image)

    # ExG
    ExG = 2 * G.astype(float) - R.astype(float) - B.astype(float)

    # ExR
    ExR = 1.4 * R.astype(float) - G.astype(float)

    # binary mask based on the (ExG - ExR >= 0)
    mask = (ExG - ExR >= 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(bbox_image, bbox_image, mask=mask)
    return result

def overlaps(x1, y1, w1, h1, bbox):
    x2, y2, x2_max, y2_max = bbox
    return not (x1 + w1 <= x2 or x2_max <= x1 or y1 + h1 <= y2 or y2_max <= y1)