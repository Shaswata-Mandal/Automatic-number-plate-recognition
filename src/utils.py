#/src/utils.py
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

def preprocess_image(image):
    """
    Preprocess the image for better plate detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    return gray, blurred

def enhance_plate_region(plate_roi):
    """
    Enhance the plate region for better OCR results
    """
    # Resize for better OCR
    plate_roi = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale if not already
    if len(plate_roi.shape) == 3:
        plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    plate_roi = cv2.adaptiveThreshold(
        plate_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    plate_roi = cv2.morphologyEx(plate_roi, cv2.MORPH_CLOSE, kernel)
    plate_roi = cv2.morphologyEx(plate_roi, cv2.MORPH_OPEN, kernel)
    
    return plate_roi

def save_processed_image(image, filename, output_dir="output"):
    """
    Save processed image to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    return output_path