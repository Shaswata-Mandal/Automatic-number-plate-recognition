#main.py
import cv2
import argparse
import os
import sys
from src.plate_detector import PlateDetector
from src.character_recognizer import CharacterRecognizer
from src.utils import save_processed_image

def process_single_image(image_path, output_dir="output"):
    """
    Process a single image for license plate recognition
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Initialize detectors
    plate_detector = PlateDetector()
    character_recognizer = CharacterRecognizer()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return
    
    print(f"Processing image: {image_path}")
    
    # Try contour-based detection first
    plate_contours, edged = plate_detector.detect_plates_contour(image)
    
    plates_found = False
    
    # Process detected plates
    for i, contour in enumerate(plate_contours):
        plate_roi, bbox = plate_detector.extract_plate_region(image, contour)
        
        if plate_roi.size == 0:
            continue
            
        # Recognize characters
        plate_text, processed_plate = character_recognizer.recognize_characters(plate_roi)
        
        if plate_text:
            plates_found = True
            print(f"Plate {i+1}: {plate_text}")
            
            # Draw bounding box and text on original image
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, plate_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save processed plate image
            plate_filename = f"plate_{os.path.basename(image_path)}_{i+1}.jpg"
            save_processed_image(processed_plate, plate_filename, output_dir)
    
    # If no plates found with contour method, try morphological method
    if not plates_found:
        print("Trying morphological detection...")
        plate_regions = plate_detector.detect_plates_morphological(image)
        
        for i, (x, y, w, h) in enumerate(plate_regions):
            plate_roi = image[y:y+h, x:x+w]
            
            if plate_roi.size == 0:
                continue
                
            # Recognize characters
            plate_text, processed_plate = character_recognizer.recognize_characters(plate_roi)
            
            if plate_text:
                plates_found = True
                print(f"Plate {i+1}: {plate_text}")
                
                # Draw bounding box and text on original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, plate_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Save processed plate image
                plate_filename = f"plate_{os.path.basename(image_path)}_{i+1}.jpg"
                save_processed_image(processed_plate, plate_filename, output_dir)
    
    if not plates_found:
        print("No license plates detected in the image.")
    
    # Save the annotated image
    annotated_filename = f"annotated_{os.path.basename(image_path)}"
    annotated_path = save_processed_image(image, annotated_filename, output_dir)
    print(f"Annotated image saved: {annotated_path}")

def process_video(video_path, output_dir="output"):
    """
    Process video for real-time license plate recognition
    """
    plate_detector = PlateDetector()
    character_recognizer = CharacterRecognizer()
    
    cap = cv2.VideoCapture(video_path if video_path != "webcam" else 0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        plate_contours, _ = plate_detector.detect_plates_contour(frame)
        
        for contour in plate_contours:
            plate_roi, bbox = plate_detector.extract_plate_region(frame, contour)
            
            if plate_roi.size > 0:
                plate_text, _ = character_recognizer.recognize_characters(plate_roi)
                
                if plate_text:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('ANPR System', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            import time
            timestamp = int(time.time())
            save_processed_image(frame, f"capture_{timestamp}.jpg", output_dir)
            print(f"Frame saved: capture_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Automatic Number Plate Recognition System')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], 
                       default='image', help='Processing mode')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if args.mode == 'image':
        if not args.input:
            print("Please provide an input image using --input parameter")
            return
        process_single_image(args.input, args.output)
    
    elif args.mode == 'video':
        if not args.input:
            print("Please provide an input video using --input parameter")
            return
        process_video(args.input, args.output)
    
    elif args.mode == 'webcam':
        process_video('webcam', args.output)

if __name__ == "__main__":
    main()