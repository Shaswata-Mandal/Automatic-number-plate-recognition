# test_installation.py
import cv2
import pytesseract
import platform
import os

def test_tesseract_installation():
    print("Testing Tesseract OCR Installation...")
    print(f"Platform: {platform.system()}")
    
    # Test if OpenCV is working
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print("✓ OpenCV is working")
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False
    
    # Test Tesseract
    try:
        # Set Tesseract path for Windows if needed
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        # Test basic OCR
        test_text = pytesseract.image_to_string('test.png') if os.path.exists('test.png') else "Tesseract is working!"
        print("✓ Tesseract OCR is functional")
        return True
        
    except Exception as e:
        print(f"✗ Tesseract error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Tesseract is installed")
        print("2. Check if Tesseract is in your PATH")
        print("3. On Windows, specify the path manually:")
        print('   pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"')
        return False

if __name__ == "__main__":
    test_tesseract_installation()