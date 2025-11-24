#/src/plate_detector.py
import cv2
import numpy as np
import imutils
from .utils import preprocess_image, save_processed_image

class PlateDetector:
    def __init__(self):
        self.min_plate_area = 1000  # Minimum area for plate region
        self.max_plate_area = 50000  # Maximum area for plate region
        
    def detect_plates_contour(self, image):
        """
        Detect license plates using contour method
        """
        gray, blurred = preprocess_image(image)
        
        # Perform edge detection
        edged = cv2.Canny(blurred, 30, 200)
        
        # Find contours in the edged image
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contours = []
        
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # If the approximated contour has 4 vertices, it might be a plate
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if self.min_plate_area < area < self.max_plate_area:
                    plate_contours.append(approx)
        
        return plate_contours, edged
    
    def detect_plates_morphological(self, image):
        """
        Detect license plates using morphological operations
        """
        gray, blurred = preprocess_image(image)
        
        # Apply morphological operations to find rectangular regions
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        dilation = cv2.dilate(blurred, rect_kernel, iterations=1)
        
        # Find contours
        contours = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            # Typical license plate aspect ratio is between 2 and 5
            if (2 < aspect_ratio < 5) and (self.min_plate_area < area < self.max_plate_area):
                plate_regions.append((x, y, w, h))
        
        return plate_regions
    
    def extract_plate_region(self, image, contour):
        """
        Extract the plate region from the image using contour
        """
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the plate region with some padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        plate_roi = image[y_start:y_end, x_start:x_end]
        return plate_roi, (x_start, y_start, x_end - x_start, y_end - y_start)