# create_test_image.py
import cv2
import numpy as np

# Create a simple test image with a mock license plate
def create_test_image():
    # Create a blank image (car background)
    image = np.ones((400, 600, 3), dtype=np.uint8) * 150  # Gray background
    
    # Add a license plate area (white rectangle)
    plate_x, plate_y = 100, 150
    plate_width, plate_height = 200, 50
    cv2.rectangle(image, (plate_x, plate_y), (plate_x + plate_width, plate_y + plate_height), 
                  (255, 255, 255), -1)
    
    # Add license plate text
    text = "DL5622"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = plate_x + (plate_width - text_size[0]) // 2
    text_y = plate_y + (plate_height + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 0), 2)
    
    # Save the test image
    cv2.imwrite('input/plate3.jpg', image)
    print("Test image created: input/test_car.jpg")
    
    return image

if __name__ == "__main__":
    create_test_image()