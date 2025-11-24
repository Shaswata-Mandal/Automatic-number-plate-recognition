# preprocess_image.py
import cv2
import numpy as np
import sys
import os

def enhance_image_for_detection(image_path):
    """Enhance image to improve license plate detection"""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return None
    
    print("🛠️ Enhancing image for better detection...")
    
    # 1. Resize if too small
    height, width = image.shape[:2]
    if width < 400:
        scale = 400 / width
        new_width = 400
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"   Resized: {width}x{height} -> {new_width}x{new_height}")
    
    # 2. Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpen image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 4. Save enhanced image
    output_path = "output/enhanced_" + os.path.basename(image_path)
    cv2.imwrite(output_path, sharpened)
    print(f"💾 Enhanced image saved: {output_path}")
    
    return sharpened

def manual_plate_selection(image_path):
    """Manually select license plate region"""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    roi = cv2.selectROI("Select License Plate Region", image)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        plate_roi = image[y:y+h, x:x+w]
        
        # Save the manually selected plate
        output_path = "output/manual_plate.jpg"
        cv2.imwrite(output_path, plate_roi)
        print(f"💾 Manually selected plate saved: {output_path}")
        
        return plate_roi
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "output/capture_1763984702.jpg"
    
    print("1. Enhancing image...")
    enhanced = enhance_image_for_detection(image_path)
    
    print("\n2. Manual selection (close the window when done)...")
    manual_plate_selection(image_path)