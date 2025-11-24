# src/character_recognizer.py
import cv2
import pytesseract
import re
import platform
import os
import numpy as np
from .utils import enhance_plate_region

class CharacterRecognizer:
    def __init__(self, tesseract_path=None):
        """
        Initialize Tesseract OCR
        """
        # Auto-detect Windows and set Tesseract path
        if platform.system() == "Windows":
            # Common Tesseract installation paths on Windows
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Tesseract found at: {path}")
                    break
            else:
                # If not found, try to use the one in PATH
                try:
                    version = pytesseract.get_tesseract_version()
                    print(f"Tesseract found in PATH: {version}")
                except:
                    print("Tesseract not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        # Use custom path if provided
        elif tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Configure Tesseract parameters for license plate recognition
        # Try different PSM modes for better accuracy
        self.tesseract_configs = [
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
    
    def preprocess_for_ocr(self, plate_image):
        """
        Preprocess plate image for better OCR accuracy
        """
        # Enhance the plate region
        enhanced = enhance_plate_region(plate_image)
        
        # Convert to grayscale if not already
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 2. Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(blurred)
        
        # 3. Apply different thresholding methods
        # Otsu's threshold
        _, thresh_otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive threshold
        thresh_adaptive = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Try both and see which works better
        # For now, return Otsu's result
        return thresh_otsu
    
    def recognize_characters(self, plate_image):
        """
        Perform OCR on the plate image with multiple config attempts
        """
        best_text = ""
        best_confidence = 0
        best_processed_image = plate_image
        
        try:
            # Preprocess the image for OCR
            processed_image = self.preprocess_for_ocr(plate_image)
            best_processed_image = processed_image
            
            # Try multiple Tesseract configurations
            for config in self.tesseract_configs:
                try:
                    # Get both text and confidence data
                    data = pytesseract.image_to_data(
                        processed_image, 
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate average confidence for non-empty words
                    confidences = [int(conf) for conf, text in zip(data['conf'], data['text']) 
                                 if int(conf) > 0 and text.strip()]
                    
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        text = ' '.join([text for text in data['text'] if text.strip()])
                        
                        # Clean the text
                        cleaned_text = self.clean_recognized_text(text)
                        
                        # Update best result if confidence is higher
                        if cleaned_text and avg_confidence > best_confidence:
                            best_text = cleaned_text
                            best_confidence = avg_confidence
                            
                except Exception as e:
                    continue
            
            # If no good results with data, try simple string method
            if not best_text:
                for config in self.tesseract_configs:
                    try:
                        text = pytesseract.image_to_string(processed_image, config=config)
                        cleaned_text = self.clean_recognized_text(text)
                        if cleaned_text:
                            best_text = cleaned_text
                            break
                    except:
                        continue
            
            return best_text, best_processed_image
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", plate_image
    
    def clean_recognized_text(self, text):
        """
        Clean and validate the recognized license plate text
        """
        # Remove special characters and whitespace, keep only alphanumeric
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Basic validation - license plates typically have specific patterns
        if self.is_valid_plate_format(cleaned):
            return cleaned
        else:
            return ""
    
    def is_valid_plate_format(self, text):
        """
        Validate if the text matches common license plate patterns
        """
        if not text:
            return False
        
        # Common license plate patterns:
        # - 6-8 alphanumeric characters
        # - At least 2 letters and 2 numbers
        if 5 <= len(text) <= 8:
            letter_count = sum(1 for c in text if c.isalpha())
            digit_count = sum(1 for c in text if c.isdigit())
            
            # Should have both letters and numbers
            if letter_count >= 2 and digit_count >= 2:
                return True
        
        return False