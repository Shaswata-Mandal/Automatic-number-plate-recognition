# manual_ocr.py
import cv2
import pytesseract
import sys
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def manual_ocr_on_plate(image_path):
    """Manually process license plate OCR"""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    # Let user select the plate region
    print("🔍 Select the license plate region and press SPACE or ENTER")
    roi = cv2.selectROI("Select License Plate", image)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = [int(i) for i in roi]
        plate_roi = image[y:y+h, x:x+w]
        
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try multiple OCR configurations
        configs = [
            '--oem 3 --psm 8',
            '--oem 3 --psm 7', 
            '--oem 3 --psm 13',
            '--oem 3 --psm 11'
        ]
        
        print("🔤 OCR Results:")
        for config in configs:
            try:
                text = pytesseract.image_to_string(thresh, config=config)
                cleaned = ''.join(c for c in text if c.isalnum()).upper()
                if len(cleaned) >= 4:
                    print(f"   Config {config}: {cleaned}")
            except:
                pass
        
        # Show processed images
        cv2.imshow('Original Plate', plate_roi)
        cv2.imshow('Processed for OCR', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save results
        cv2.imwrite('output/manual_plate_roi.jpg', plate_roi)
        cv2.imwrite('output/manual_plate_processed.jpg', thresh)
        print("💾 Results saved to output/ folder")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "output/capture_1763984702.jpg"
    
    manual_ocr_on_plate(image_path)