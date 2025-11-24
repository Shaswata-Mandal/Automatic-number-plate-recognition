🚗 Automatic Number Plate Recognition (ANPR) System

A Python-based computer vision system for automatically detecting and recognizing license plates from images and video streams.


📌 Features:

-> Multi-format Input Support: Images, video files, and live webcam feeds

-> Dual Detection Strategy: Contour-based + morphological operations

-> Real-time Processing: 15–18 FPS on video streams

-> High Accuracy: ~85% plate detection, ~82% character recognition

-> Modular & Extensible: Easy to integrate new algorithms or OCR engines

🛠️ Tech Stack

-> Python

-> OpenCV – Image processing and plate detection

-> Tesseract OCR – Character recognition


🚀 Usage:

# Process an image
python main.py --mode image --input path/to/image.jpg

# Process a video
python main.py --mode video --input path/to/video.mp4

# Use webcam
python main.py --mode webcam

🧪 Testing
Run the test suite to validate installation and functionality:

-> python test_installation.py


🧠 Future Enhancements:

-> Deep Learning (YOLO/CNN) for better accuracy

-> GPU acceleration

-> Multi-threading & cloud deployment

-> Support for international plate formats


👨‍💻 Author:

Shaswata Mandal
VIT Bhopal – 23BHI10018
Computer Vision Project | Interim Semester 2025–26